from __future__ import annotations

import csv
import hmac
import io
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from uuid import uuid4

from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

from .benchmarking import local_cost_scenarios
from .config import settings
from .datasets import build_dataset, diff_datasets, get_dataset, list_datasets
from .db import asdict, init_db, transaction
from .evaluation import run_mlx_evaluation
from .evaluation_sets import create_evaluation_set, get_evaluation_set, list_evaluation_sets
from .feedback import (
    delete_feedback,
    get_feedback,
    import_records,
    list_feedback,
    redact_feedback,
    resolve_conflict,
    review,
)
from .inference import predict
from .registry import active_model, deployments, gate, list_models, promote, rollback
from .tasks.products import ProductAttributes, ProductExtractionTask
from .training import cancel_job, get_job, list_jobs, mlx_status, reconcile_jobs, start_training


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    init_db()
    reconcile_jobs()
    yield


app = FastAPI(
    title="LLM Feedback Fine-Tuning Pipeline API",
    version="0.1.0",
    description="Local correction-to-model workflow",
    lifespan=lifespan,
)


class PredictionRequest(BaseModel):
    input_text: str = Field(min_length=1, max_length=20_000)
    mode: str | None = None


class FeedbackRequest(BaseModel):
    input_text: str = Field(min_length=1, max_length=20_000)
    corrected_output: ProductAttributes
    original_prediction: dict[str, Any] | None = None
    source: str = "api"
    source_record_id: str | None = None
    source_group_id: str | None = None
    original_model_id: str | None = None
    prompt_version: str | None = None
    model_generation_metadata: dict[str, Any] | None = None
    notes: str | None = None


class ReviewRequest(BaseModel):
    decision: str
    reviewer: str = "local-operator"
    notes: str | None = None


class BulkReviewRequest(BaseModel):
    feedback_ids: list[str] = Field(min_length=1, max_length=500)
    decision: str
    reviewer: str = "local-operator"
    notes: str | None = None
    confirm: bool = False


class RedactRequest(BaseModel):
    reviewer: str = "local-operator"


class ResolveConflictRequest(BaseModel):
    reviewer: str = "local-operator"
    reason: str = Field(min_length=1, max_length=2_000)


class DeleteFeedbackRequest(BaseModel):
    operator: str = Field(min_length=1, max_length=200)
    reason: str = Field(min_length=1, max_length=2_000)


class BuildRequest(BaseModel):
    seed: int = 42
    train_pct: float = Field(default=0.8, ge=0, le=1)
    valid_pct: float = Field(default=0.1, ge=0, le=1)
    test_pct: float = Field(default=0.1, ge=0, le=1)


class TrainingRequest(BaseModel):
    dataset_id: str
    preset: str = "smoke"
    model_id: str | None = None


class EvaluationRequest(BaseModel):
    dataset_id: str
    candidate_id: str
    evaluation_set_id: str | None = None


class CostRequest(BaseModel):
    accuracy: float = Field(ge=0, le=1)
    power_watts: float | None = Field(default=None, gt=0)
    electricity_usd_per_kwh: float | None = Field(default=None, ge=0)
    inference_seconds: float | None = Field(default=None, ge=0)
    training_cost_usd: float | None = Field(default=None, ge=0)
    monthly_hardware_cost_usd: float | None = Field(default=None, ge=0)
    training_amortization_months: int = Field(default=12, ge=1)


class PromotionRequest(BaseModel):
    operator: str = Field(default="local-operator", min_length=1, max_length=200)
    reason: str = Field(min_length=1, max_length=2_000)
    override: bool = False


def admin(x_llm_feedback_key: str | None = Header(default=None, alias="X-LLM-Feedback-Key")) -> None:
    if settings.api_key and not hmac.compare_digest(x_llm_feedback_key or "", settings.api_key):
        raise HTTPException(status_code=401, detail={"error": "unauthorized"})


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict[str, Any]:
    return {"database": "ready", "model": active_model(), "mlx": mlx_status()}


@app.get("/api/v1/tasks")
def tasks() -> list[dict[str, Any]]:
    return [
        {
            "task_id": ProductExtractionTask.task_id,
            "schema_version": ProductExtractionTask.schema_version,
            "output_schema": ProductAttributes.model_json_schema(),
            "system_prompt": ProductExtractionTask.system_prompt,
        }
    ]


@app.post("/api/v1/tasks/{task_id}/predict")
def api_predict(task_id: str, request: PredictionRequest) -> dict[str, Any]:
    if task_id != ProductExtractionTask.task_id:
        raise HTTPException(404, detail={"error": "unknown_task"})
    selected_mode = request.mode or ("simulated" if settings.allow_simulated else "mlx")
    if selected_mode == "mlx" and active_model() is None:
        raise HTTPException(
            503,
            detail={"error": "no_active_model", "message": "Promote a healthy local model before inference."},
        )
    try:
        return {"request_id": str(uuid4()), **predict(request.input_text, mode=selected_mode)}
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(503, detail={"error": "inference_failed", "message": str(exc)}) from exc


@app.post("/api/v1/feedback", dependencies=[Depends(admin)])
def create_feedback(request: FeedbackRequest) -> dict[str, Any]:
    record = request.model_dump(mode="json")
    result = import_records([record], commit=True)
    if result["invalid"]:
        raise HTTPException(422, detail=result["invalid"])
    with transaction() as con:
        row = con.execute("SELECT * FROM feedback WHERE id=?", (result["feedback_ids"][0],)).fetchone()
    return {"import": result, "feedback": asdict(row)}


@app.get("/api/v1/feedback", dependencies=[Depends(admin)])
def feedback(
    status: str | None = Query(default=None),
    source: str | None = None,
    task_id: str | None = None,
    created_after: str | None = None,
    created_before: str | None = None,
) -> list[dict[str, Any]]:
    return list_feedback(
        status, source=source, task_id=task_id, created_after=created_after, created_before=created_before
    )


@app.get("/api/v1/feedback/{feedback_id}", dependencies=[Depends(admin)])
def feedback_detail(feedback_id: str) -> dict[str, Any]:
    try:
        return get_feedback(feedback_id)
    except KeyError as exc:
        raise HTTPException(404, detail={"error": "feedback_not_found"}) from exc


@app.post("/api/v1/feedback/{feedback_id}/review", dependencies=[Depends(admin)])
def review_feedback(feedback_id: str, request: ReviewRequest) -> dict[str, Any]:
    try:
        return review(feedback_id, request.decision, request.reviewer, request.notes)
    except KeyError as exc:
        raise HTTPException(404, detail={"error": "feedback_not_found"}) from exc
    except ValueError as exc:
        raise HTTPException(409, detail={"error": "review_conflict", "message": str(exc)}) from exc


@app.post("/api/v1/feedback/{feedback_id}/redact", dependencies=[Depends(admin)])
def redact(feedback_id: str, request: RedactRequest) -> dict[str, Any]:
    try:
        return redact_feedback(feedback_id, request.reviewer)
    except KeyError as exc:
        raise HTTPException(404, detail={"error": "feedback_not_found"}) from exc
    except ValueError as exc:
        raise HTTPException(409, detail={"error": "redaction_unavailable", "message": str(exc)}) from exc


@app.post("/api/v1/feedback/{feedback_id}/resolve-conflict", dependencies=[Depends(admin)])
def resolve_feedback_conflict(feedback_id: str, request: ResolveConflictRequest) -> dict[str, Any]:
    try:
        return resolve_conflict(feedback_id, request.reviewer, request.reason)
    except KeyError as exc:
        raise HTTPException(404, detail={"error": "feedback_not_found"}) from exc
    except ValueError as exc:
        raise HTTPException(409, detail={"error": "conflict_resolution_failed", "message": str(exc)}) from exc


@app.delete("/api/v1/feedback/{feedback_id}", dependencies=[Depends(admin)])
def delete_feedback_record(feedback_id: str, request: DeleteFeedbackRequest) -> dict[str, Any]:
    try:
        return delete_feedback(feedback_id, request.operator, request.reason)
    except KeyError as exc:
        raise HTTPException(404, detail={"error": "feedback_not_found"}) from exc
    except ValueError as exc:
        raise HTTPException(409, detail={"error": "feedback_delete_blocked", "message": str(exc)}) from exc


@app.post("/api/v1/feedback/review/bulk", dependencies=[Depends(admin)])
def bulk_review(request: BulkReviewRequest) -> dict[str, Any]:
    if not request.confirm:
        raise HTTPException(409, detail={"error": "explicit_bulk_confirmation_required"})
    reviewed, errors = [], []
    for feedback_id in request.feedback_ids:
        try:
            reviewed.append(review(feedback_id, request.decision, request.reviewer, request.notes))
        except (KeyError, ValueError) as exc:
            errors.append({"id": feedback_id, "error": str(exc)})
    return {"reviewed": reviewed, "errors": errors}


@app.post("/api/v1/datasets/build", dependencies=[Depends(admin)])
def create_dataset(request: BuildRequest) -> dict[str, Any]:
    try:
        return build_dataset(**request.model_dump())
    except ValueError as exc:
        raise HTTPException(422, detail={"error": str(exc)}) from exc


@app.get("/api/v1/datasets", dependencies=[Depends(admin)])
def datasets() -> list[dict[str, Any]]:
    return list_datasets()


@app.get("/api/v1/datasets/{dataset_id}", dependencies=[Depends(admin)])
def dataset_detail(dataset_id: str) -> dict[str, Any]:
    try:
        return get_dataset(dataset_id)
    except KeyError as exc:
        raise HTTPException(404, detail={"error": "dataset_not_found"}) from exc


@app.get("/api/v1/datasets/{dataset_id}/diff/{other_id}", dependencies=[Depends(admin)])
def dataset_diff(dataset_id: str, other_id: str) -> dict[str, Any]:
    try:
        return diff_datasets(dataset_id, other_id)
    except KeyError as exc:
        raise HTTPException(404, detail={"error": "dataset_not_found"}) from exc


@app.post("/api/v1/train/jobs", dependencies=[Depends(admin)])
def train(request: TrainingRequest) -> dict[str, Any]:
    try:
        return start_training(request.dataset_id, request.preset, request.model_id)
    except KeyError as exc:
        raise HTTPException(404, detail={"error": "dataset_not_found"}) from exc
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(503, detail={"error": "training_unavailable", "message": str(exc)}) from exc


@app.get("/api/v1/experiments", dependencies=[Depends(admin)])
def experiments() -> list[dict[str, Any]]:
    return list_jobs()


@app.get("/api/v1/train/jobs/{experiment_id}", dependencies=[Depends(admin)])
def train_job(experiment_id: str) -> dict[str, Any]:
    try:
        return get_job(experiment_id)
    except KeyError as exc:
        raise HTTPException(404, detail={"error": "job_not_found"}) from exc


@app.post("/api/v1/train/jobs/{experiment_id}/cancel", dependencies=[Depends(admin)])
def train_cancel(experiment_id: str) -> dict[str, Any]:
    try:
        return cancel_job(experiment_id)
    except KeyError as exc:
        raise HTTPException(404, detail={"error": "job_not_found"}) from exc
    except ValueError as exc:
        raise HTTPException(409, detail={"error": "job_not_cancellable", "message": str(exc)}) from exc


@app.get("/api/v1/models", dependencies=[Depends(admin)])
def models() -> list[dict[str, Any]]:
    return list_models()


@app.get("/api/v1/models/active", dependencies=[Depends(admin)])
def model_active() -> dict[str, Any] | None:
    return active_model()


@app.get("/api/v1/models/{model_id}/promotion-check")
def promotion_check(model_id: str) -> dict[str, Any]:
    for item in list_models():
        if item["id"] == model_id:
            return gate(item)
    raise HTTPException(404, detail={"error": "model_not_found"})


@app.post("/api/v1/models/{model_id}/promote", dependencies=[Depends(admin)])
def promote_model(model_id: str, request: PromotionRequest) -> dict[str, Any]:
    try:
        return promote(model_id, request.operator, request.reason, override=request.override)
    except KeyError as exc:
        raise HTTPException(404, detail={"error": "model_not_found"}) from exc
    except ValueError as exc:
        raise HTTPException(409, detail={"error": "promotion_blocked", "message": str(exc)}) from exc


@app.post("/api/v1/models/rollback", dependencies=[Depends(admin)])
def rollback_model(request: PromotionRequest) -> dict[str, Any]:
    try:
        return rollback(request.operator, request.reason)
    except ValueError as exc:
        raise HTTPException(409, detail={"error": "rollback_unavailable", "message": str(exc)}) from exc


@app.get("/api/v1/deployments")
def deployment_history() -> list[dict[str, Any]]:
    return deployments()


@app.get("/api/v1/evaluations", dependencies=[Depends(admin)])
def evaluations() -> list[dict[str, Any]]:
    with transaction() as con:
        return [
            {**dict(row), "report": json.loads(row["report"])}
            for row in con.execute("SELECT * FROM evaluations ORDER BY created_at DESC")
        ]


@app.post("/api/v1/evaluation-sets", dependencies=[Depends(admin)])
async def freeze_evaluation_set(file: UploadFile = File(...)) -> dict[str, Any]:  # noqa: B008
    try:
        name = (file.filename or "").lower()
        content = (await file.read(10_000_001)).decode("utf-8-sig")
        if len(content) > 10_000_000:
            raise ValueError("Evaluation-set upload exceeds the 10 MB limit")
        if name.endswith(".csv"):
            records = list(csv.DictReader(io.StringIO(content)))
        elif name.endswith((".jsonl", ".ndjson")):
            records = []
            for line, row in enumerate(content.splitlines(), 1):
                if row.strip():
                    try:
                        records.append(json.loads(row))
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"line {line}: invalid JSON: {exc.msg}") from exc
        else:
            raise ValueError("Upload a CSV or JSONL evaluation file")
        return create_evaluation_set(records)
    except ValueError as exc:
        raise HTTPException(422, detail={"error": str(exc)}) from exc


@app.get("/api/v1/evaluation-sets", dependencies=[Depends(admin)])
def evaluation_sets() -> list[dict[str, Any]]:
    return list_evaluation_sets()


@app.get("/api/v1/evaluation-sets/{evaluation_set_id}", dependencies=[Depends(admin)])
def evaluation_set_detail(evaluation_set_id: str) -> dict[str, Any]:
    try:
        return get_evaluation_set(evaluation_set_id)
    except KeyError as exc:
        raise HTTPException(404, detail={"error": "evaluation_set_not_found"}) from exc


@app.get("/api/v1/evaluations/{evaluation_id}", dependencies=[Depends(admin)])
def evaluation_detail(evaluation_id: str) -> dict[str, Any]:
    with transaction() as con:
        row = con.execute("SELECT * FROM evaluations WHERE id=?", (evaluation_id,)).fetchone()
    if row is None:
        raise HTTPException(404, detail={"error": "evaluation_not_found"})
    return {**dict(row), "report": json.loads(row["report"])}


@app.post("/api/v1/benchmarks/cost", dependencies=[Depends(admin)])
def benchmark_cost(request: CostRequest) -> dict[str, Any]:
    return local_cost_scenarios(**request.model_dump())


@app.post("/api/v1/evaluations", dependencies=[Depends(admin)])
def create_evaluation(request: EvaluationRequest) -> dict[str, Any]:
    try:
        return run_mlx_evaluation(request.dataset_id, request.candidate_id, request.evaluation_set_id)
    except KeyError as exc:
        raise HTTPException(404, detail={"error": "dataset_or_candidate_not_found"}) from exc
    except (ImportError, RuntimeError, ValueError) as exc:
        raise HTTPException(503, detail={"error": "evaluation_unavailable", "message": str(exc)}) from exc


@app.post("/api/v1/import/preview", dependencies=[Depends(admin)])
async def import_preview(  # noqa: B008
    file: UploadFile = File(...),  # noqa: B008
    commit: bool = False,
) -> dict[str, Any]:
    try:
        name = (file.filename or "").lower()
        content = (await file.read(10_000_001)).decode("utf-8-sig")
        if len(content) > 10_000_000:
            raise ValueError("Import file exceeds the 10 MB limit")
        if name.endswith(".csv"):
            records = list(csv.DictReader(io.StringIO(content)))
        elif name.endswith((".jsonl", ".ndjson")):
            records = []
            for line, row in enumerate(content.splitlines(), 1):
                if row.strip():
                    try:
                        records.append(json.loads(row))
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"line {line}: invalid JSON: {exc.msg}") from exc
        else:
            raise ValueError("Upload a CSV or JSONL file")
        return import_records(records, commit=commit)
    except ValueError as exc:
        raise HTTPException(422, detail={"error": str(exc)}) from exc
