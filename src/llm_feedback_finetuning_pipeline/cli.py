from __future__ import annotations

import json
import platform
import shutil
from pathlib import Path

import typer
import uvicorn

from . import __version__
from .config import settings
from .datasets import build_dataset, diff_datasets, get_dataset, list_datasets
from .db import init_db
from .evaluation import run_mlx_evaluation
from .evaluation_sets import create_evaluation_set, get_evaluation_set, list_evaluation_sets
from .feedback import delete_feedback, import_records, list_feedback, read_import, redact_feedback, review
from .inference import predict
from .registry import active_model, list_models, promote, rollback
from .training import cancel_job, get_job, list_jobs, mlx_status, start_training

app = typer.Typer(help="Turn reviewed business corrections into specialized local models.")
feedback_app = typer.Typer(help="Import and review corrections.")
dataset_app = typer.Typer(help="Build immutable dataset snapshots.")
model_app = typer.Typer(help="Inspect and deploy model candidates.")
train_app = typer.Typer(help="Run local MLX-LM LoRA training.")
evaluate_app = typer.Typer(help="Run paired model evaluations.")
evaluation_set_app = typer.Typer(help="Manage separate frozen evaluation holdouts.")
app.add_typer(feedback_app, name="feedback")
app.add_typer(dataset_app, name="dataset")
app.add_typer(model_app, name="models")
app.add_typer(train_app, name="train")
app.add_typer(evaluate_app, name="evaluate")
app.add_typer(evaluation_set_app, name="evaluation-set")


def output(value: object, as_json: bool = True) -> None:
    typer.echo(json.dumps(value, indent=2, default=str) if as_json else str(value))


@app.command()
def init() -> None:
    """Initialize the database and artifact directories."""
    init_db()
    output(
        {
            "status": "initialized",
            "home": str(settings.data_dir),
            "database": str(settings.database_path),
        }
    )


@app.command()
def doctor() -> None:
    """Check Python, database, disk, and optional MLX readiness."""
    settings.initialize()
    init_db()
    disk = shutil.disk_usage(settings.data_dir)
    report = {
        "version": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "data_dir": str(settings.data_dir),
        "database_initialized": settings.database_path.exists(),
        "free_disk_gb": round(disk.free / (1024**3), 2),
        "mlx": mlx_status(),
    }
    output(report)


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Start the local API. Defaults to localhost only."""
    if host not in {"127.0.0.1", "localhost", "::1"}:
        typer.echo("Warning: binding beyond localhost exposes this service to the network.", err=True)
    init_db()
    uvicorn.run("llm_feedback_finetuning_pipeline.api:app", host=host, port=port, reload=False)


@app.command("predict")
def predict_command(
    text: str = typer.Option(..., "--text"),
    mode: str = typer.Option("simulated", help="simulated demo or real mlx inference"),
) -> None:
    """Run a product extraction; choose mlx for real local model inference."""
    try:
        output(predict(text, mode=mode))
    except (RuntimeError, ValueError) as exc:
        typer.echo(f"Inference failed: {exc}", err=True)
        raise typer.Exit(1) from exc


@feedback_app.command("import")
def import_file(file: Path, commit: bool = False) -> None:
    """Preview CSV/JSONL correction imports; use --commit after reviewing the preview."""
    init_db()
    result = import_records(read_import(file), commit=commit)
    output(result)
    if result["invalid"]:
        raise typer.Exit(2)


@feedback_app.command("list")
def feedback_list(status: str | None = None) -> None:
    init_db()
    output(list_feedback(status))


@feedback_app.command("review")
def feedback_review(
    feedback_id: str, decision: str, reviewer: str = "local-operator", notes: str | None = None
) -> None:
    init_db()
    output(review(feedback_id, decision, reviewer, notes))


@feedback_app.command("redact")
def feedback_redact(feedback_id: str, reviewer: str = "local-operator") -> None:
    init_db()
    output(redact_feedback(feedback_id, reviewer))


@feedback_app.command("delete")
def feedback_delete(feedback_id: str, reason: str, operator: str = "local-operator") -> None:
    """Delete feedback that is not referenced by an immutable dataset snapshot."""
    init_db()
    output(delete_feedback(feedback_id, operator, reason))


@dataset_app.command("build")
def dataset_build(seed: int = 42, train_pct: float = 0.8, valid_pct: float = 0.1, test_pct: float = 0.1) -> None:
    init_db()
    output(build_dataset(seed=seed, train_pct=train_pct, valid_pct=valid_pct, test_pct=test_pct))


@dataset_app.command("list")
def dataset_list() -> None:
    init_db()
    output(list_datasets())


@dataset_app.command("inspect")
def dataset_inspect(dataset_id: str) -> None:
    init_db()
    output(get_dataset(dataset_id))


@dataset_app.command("diff")
def dataset_diff(old_dataset_id: str, new_dataset_id: str) -> None:
    init_db()
    output(diff_datasets(old_dataset_id, new_dataset_id))


@train_app.command("start")
def train_start(dataset_id: str, preset: str = "smoke", model_id: str | None = None) -> None:
    init_db()
    output(start_training(dataset_id, preset, model_id))


@train_app.command("status")
def train_status(experiment_id: str | None = typer.Argument(None)) -> None:
    init_db()
    if experiment_id:
        output(get_job(experiment_id))
    else:
        output(list_jobs())


@train_app.command("cancel")
def train_cancel(experiment_id: str) -> None:
    init_db()
    output(cancel_job(experiment_id))


@evaluate_app.command("run")
def evaluate_run(
    dataset_id: str,
    candidate_id: str,
    evaluation_set_id: str | None = typer.Option(None, "--evaluation-set"),
) -> None:
    """Evaluate a candidate on a dataset test split or an independent frozen holdout."""
    init_db()
    output(run_mlx_evaluation(dataset_id, candidate_id, evaluation_set_id))


@evaluation_set_app.command("freeze")
def evaluation_set_freeze(file: Path) -> None:
    init_db()
    output(create_evaluation_set(read_import(file)))


@evaluation_set_app.command("list")
def evaluation_set_list() -> None:
    init_db()
    output(list_evaluation_sets())


@evaluation_set_app.command("inspect")
def evaluation_set_inspect(evaluation_set_id: str) -> None:
    init_db()
    output(get_evaluation_set(evaluation_set_id))


@model_app.command("list")
def models_list() -> None:
    init_db()
    output({"active": active_model(), "models": list_models()})


@model_app.command("promote")
def models_promote(model_id: str, reason: str, operator: str = "local-operator", override: bool = False) -> None:
    init_db()
    output(promote(model_id, operator, reason, override=override))


@model_app.command("rollback")
def models_rollback(reason: str, operator: str = "local-operator") -> None:
    init_db()
    output(rollback(operator, reason))


@app.command()
def demo() -> None:
    """Load a clearly synthetic, pre-reviewed catalog correction fixture."""
    init_db()
    from .demo_data import seed_demo

    output(seed_demo())


if __name__ == "__main__":
    app()
