from __future__ import annotations

import hashlib
import json
from typing import Any

from .db import asdict, new_id, now, path_for, transaction


def adapter_checksum(adapter: Any) -> str:
    digest = hashlib.sha256()
    files = list(adapter.glob("*.safetensors"))
    config = adapter / "adapter_config.json"
    if config.is_file():
        files.append(config)
    for artifact in sorted(files):
        digest.update(artifact.name.encode())
        digest.update(artifact.read_bytes())
    return digest.hexdigest()


def register_model(
    base_model: str,
    *,
    adapter_path: str | None = None,
    dataset_id: str | None = None,
    experiment_id: str | None = None,
    evaluation_id: str | None = None,
    metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    model_id = new_id("model")
    if adapter_path:
        path_for("models", adapter_path)
    with transaction() as con:
        con.execute(
            "INSERT INTO models(id,base_model,adapter_path,dataset_id,experiment_id,evaluation_id,"
            "status,metrics,created_at) VALUES(?,?,?,?,?,?,?,?,?)",
            (
                model_id,
                base_model,
                adapter_path,
                dataset_id,
                experiment_id,
                evaluation_id,
                "candidate",
                json.dumps(metrics or {}),
                now(),
            ),
        )
        return asdict(con.execute("SELECT * FROM models WHERE id=?", (model_id,)).fetchone()) or {}


def list_models() -> list[dict[str, Any]]:
    with transaction() as con:
        return [asdict(row) or {} for row in con.execute("SELECT * FROM models ORDER BY created_at DESC")]


def active_model() -> dict[str, Any] | None:
    with transaction() as con:
        row = con.execute("SELECT value FROM app_state WHERE key='active_model'").fetchone()
        if not row:
            return None
        model = con.execute("SELECT * FROM models WHERE id=?", (row[0],)).fetchone()
        return asdict(model)


def gate(model: dict[str, Any], min_samples: int = 30) -> dict[str, Any]:
    metrics = model.get("metrics") or {}
    sample = metrics.get("n", 0)
    report: dict[str, Any] = {}
    if model.get("evaluation_id"):
        with transaction() as con:
            evaluation = con.execute("SELECT report FROM evaluations WHERE id=?", (model["evaluation_id"],)).fetchone()
        if evaluation:
            report = json.loads(evaluation["report"])
    baseline_accuracy = (report.get("baseline") or {}).get("exact_match_accuracy")
    candidate_accuracy = metrics.get("exact_match_accuracy")
    regression_rate = report.get("regression_rate_of_baseline_correct")
    fields = metrics.get("per_field_accuracy") or {}
    critical_fields = {name: fields.get(name) for name in ("category", "size", "material")}
    latency = metrics.get("latency_ms_p95")
    peak_rss = report.get("observed_process_peak_rss_bytes")
    artifact_valid = False
    if model.get("adapter_path"):
        try:
            adapter = path_for("models", model["adapter_path"])
            artifact_valid = (
                adapter.is_dir()
                and any(adapter.glob("*.safetensors"))
                and adapter_checksum(adapter) == metrics.get("adapter_sha256")
            )
        except (OSError, ValueError):
            artifact_valid = False
    lineage_valid = bool(report) and report.get("dataset_id") == model.get("dataset_id")
    checks = {
        "evaluation_sample": {
            "pass": sample >= min_samples,
            "observed": sample,
            "required": min_samples,
        },
        "schema_adherence": {
            "pass": metrics.get("schema_adherence", 0) >= 0.98,
            "observed": metrics.get("schema_adherence"),
            "required": 0.98,
        },
        "evaluation_attached": {
            "pass": bool(model.get("evaluation_id")),
            "observed": model.get("evaluation_id"),
        },
        "evaluation_lineage": {
            "pass": lineage_valid,
            "observed": report.get("dataset_id"),
            "required": model.get("dataset_id"),
        },
        "fresh_evaluation_holdout": {
            "pass": report.get("evaluation_reuse_count", 0) == 0,
            "observed_prior_runs": report.get("evaluation_reuse_count", 0),
            "required": "first evaluation against this frozen test set",
        },
        "accuracy_improved": {
            "pass": baseline_accuracy is not None
            and candidate_accuracy is not None
            and candidate_accuracy > baseline_accuracy,
            "observed_delta": candidate_accuracy - baseline_accuracy
            if baseline_accuracy is not None and candidate_accuracy is not None
            else None,
            "required": "strict improvement over the paired untuned baseline",
        },
        "regression_budget": {
            "pass": regression_rate is not None and regression_rate <= 0.05,
            "observed": regression_rate,
            "required_maximum": 0.05,
        },
        "critical_field_floor": {
            "pass": all(value is not None and value >= 0.70 for value in critical_fields.values()),
            "observed": critical_fields,
            "required_minimum": 0.70,
        },
        "latency_budget": {
            "pass": latency is not None and latency <= 5000,
            "observed_p95_ms": latency,
            "required_maximum_ms": 5000,
        },
        "memory_budget": {
            "pass": peak_rss is not None and peak_rss <= 48 * 1024**3,
            "observed_process_peak_rss_bytes": peak_rss,
            "required_maximum_bytes": 48 * 1024**3,
        },
        "adapter_integrity": {
            "pass": artifact_valid,
            "observed": "verified checksum and weights" if artifact_valid else "missing or invalid adapter artifact",
        },
    }
    return {"eligible": all(v["pass"] for v in checks.values()), "checks": checks}


def promote(model_id: str, operator: str, reason: str, *, override: bool = False) -> dict[str, Any]:
    with transaction() as con:
        row = con.execute("SELECT * FROM models WHERE id=?", (model_id,)).fetchone()
    if not row:
        raise KeyError(model_id)
    model = asdict(row) or {}
    decision = gate(model)
    if not decision["eligible"] and not override:
        raise ValueError(json.dumps({"decision": "blocked", **decision}))
    if model.get("adapter_path"):
        adapter = path_for("models", model["adapter_path"])
        if not adapter.exists() or not any(adapter.glob("*.safetensors")):
            raise ValueError("Candidate adapter directory or weights are missing")
        expected_checksum = (model.get("metrics") or {}).get("adapter_sha256")
        if not expected_checksum or adapter_checksum(adapter) != expected_checksum:
            raise ValueError("Candidate adapter checksum is missing or does not match its registered artifact")
    with transaction() as con:
        previous = con.execute("SELECT value FROM app_state WHERE key='active_model'").fetchone()
    old_id = previous[0] if previous else None
    from .inference import predict

    predict(
        "Health check product with no stated attributes.",
        mode="mlx",
        model_id=model_id,
        base_model=model["base_model"],
        adapter_path=model.get("adapter_path"),
    )
    with transaction() as con:
        previous = con.execute("SELECT value FROM app_state WHERE key='active_model'").fetchone()
        live_old_id = previous[0] if previous else None
        if live_old_id != old_id:
            raise ValueError("Active deployment changed during promotion health check; retry")
        if old_id:
            con.execute("UPDATE models SET status='archived' WHERE id=?", (old_id,))
        con.execute("UPDATE models SET status='production' WHERE id=?", (model_id,))
        con.execute(
            "INSERT INTO app_state(key,value) VALUES('active_model',?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (model_id,),
        )
        con.execute(
            "INSERT INTO deployments(id,model_id,action,operator,reason,created_at,details) VALUES(?,?,?,?,?,?,?)",
            (
                new_id("dep"),
                model_id,
                "promote_override" if override and not decision["eligible"] else "promote",
                operator,
                reason,
                now(),
                json.dumps(
                    {
                        "previous_model": old_id,
                        "promotion_gates": decision,
                        "failed_checks": [key for key, check in decision["checks"].items() if not check["pass"]],
                        "override": override and not decision["eligible"],
                    }
                ),
            ),
        )
    return {
        "active_model": model_id,
        "previous_model": old_id,
        "decision": decision,
        "override": override and not decision["eligible"],
    }


def rollback(operator: str, reason: str) -> dict[str, Any]:
    with transaction() as con:
        current = con.execute("SELECT value FROM app_state WHERE key='active_model'").fetchone()
        prior = con.execute(
            "SELECT model_id FROM deployments WHERE action IN ('promote','promote_override','rollback') "
            "ORDER BY created_at DESC LIMIT 1 OFFSET 1"
        ).fetchone()
    if not current or not prior:
        raise ValueError("No previous deployment is available to roll back to")
    current_id, target = current[0], prior[0]
    with transaction() as con:
        row = con.execute("SELECT * FROM models WHERE id=?", (target,)).fetchone()
    if not row:
        raise ValueError("Previous model registry record is missing")
    model = asdict(row) or {}
    if model.get("adapter_path"):
        adapter = path_for("models", model["adapter_path"])
        if not adapter.exists() or not any(adapter.glob("*.safetensors")):
            raise ValueError("Previous model adapter files are missing")
        expected_checksum = (model.get("metrics") or {}).get("adapter_sha256")
        if not expected_checksum or adapter_checksum(adapter) != expected_checksum:
            raise ValueError("Previous model adapter checksum is missing or no longer matches its artifact")
    from .inference import predict

    predict(
        "Health check product with no stated attributes.",
        mode="mlx",
        model_id=target,
        base_model=model["base_model"],
        adapter_path=model.get("adapter_path"),
    )
    with transaction() as con:
        live_current = con.execute("SELECT value FROM app_state WHERE key='active_model'").fetchone()
        if not live_current or live_current[0] != current_id:
            raise ValueError("Active deployment changed during rollback health check; retry")
        con.execute("UPDATE models SET status='archived' WHERE id=?", (current_id,))
        con.execute("UPDATE models SET status='production' WHERE id=?", (target,))
        con.execute("UPDATE app_state SET value=? WHERE key='active_model'", (target,))
        con.execute(
            "INSERT INTO deployments(id,model_id,action,operator,reason,created_at,details) VALUES(?,?,?,?,?,?,?)",
            (
                new_id("dep"),
                target,
                "rollback",
                operator,
                reason,
                now(),
                json.dumps({"previous_model": current[0], "health_check": "passed"}),
            ),
        )
    return {"active_model": target, "previous_model": current[0], "status": "rolled_back"}


def deployments() -> list[dict[str, Any]]:
    with transaction() as con:
        return [
            {**dict(row), "details": json.loads(row["details"])}
            for row in con.execute("SELECT * FROM deployments ORDER BY created_at DESC")
        ]
