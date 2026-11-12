from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

from .config import settings
from .datasets import verify_dataset_artifacts
from .db import new_id, now, path_for, transaction
from .tasks.products import ProductExtractionTask

DEFAULT_MODEL = "Qwen/Qwen3-0.6B-MLX-bf16"


def model_is_available(model_ref: str) -> bool:
    local = Path(model_ref).expanduser()
    if local.exists():
        return True
    try:
        from huggingface_hub import try_to_load_from_cache

        cached = try_to_load_from_cache(model_ref, "config.json")
        return isinstance(cached, str) and Path(cached).is_file()
    except Exception:
        return False


def cached_model_revision(model_ref: str) -> str | None:
    """Return the local Hugging Face snapshot commit when it can be resolved offline."""
    local = Path(model_ref).expanduser()
    if local.exists():
        return None
    try:
        from huggingface_hub import try_to_load_from_cache

        config_file = try_to_load_from_cache(model_ref, "config.json")
        if isinstance(config_file, str) and Path(config_file).is_file():
            return Path(config_file).parent.name
    except Exception:
        return None
    return None


def mlx_status() -> dict[str, Any]:
    import psutil

    result: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "available": False,
        "available_memory_gb": round(psutil.virtual_memory().available / 1024**3, 2),
        "free_disk_gb": round(shutil.disk_usage(settings.data_dir.parent).free / 1024**3, 2),
        "configured_model": settings.model_id,
        "model_available": model_is_available(settings.model_id),
    }
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        result["reason"] = "MLX requires Apple Silicon macOS for this project"
        return result
    try:
        import mlx  # noqa: F401

        mx = importlib.import_module("mlx.core")

        result["mlx_version"] = importlib.metadata.version("mlx")
        result["mlx_lm_version"] = importlib.metadata.version("mlx-lm")
        result["default_device"] = str(mx.default_device())
        result["available"] = True
    except importlib.metadata.PackageNotFoundError as exc:
        result["reason"] = f"MLX packages are not installed: {exc}"
    except ImportError as exc:
        result["reason"] = f"MLX backend could not initialize: {exc}"
    return result


def start_training(dataset_id: str, preset: str = "smoke", model_id: str | None = None) -> dict[str, Any]:
    verify_dataset_artifacts(dataset_id)
    mlx = mlx_status()
    if not mlx["available"]:
        raise RuntimeError(str(mlx.get("reason", "MLX is unavailable")))
    if mlx["available_memory_gb"] < 8:
        raise RuntimeError(
            f"Only {mlx['available_memory_gb']} GB of system memory is currently available; "
            "free memory and retry. This estimate does not account for every MLX allocation."
        )
    if mlx["free_disk_gb"] < 10:
        raise RuntimeError(
            f"Only {mlx['free_disk_gb']} GB of free disk is available; model weights and artifacts need space."
        )
    presets = {
        "smoke": {"iters": 20, "batch_size": 1, "learning_rate": 1e-5, "val_batches": 2},
        "standard": {"iters": 500, "batch_size": 2, "learning_rate": 1e-5, "val_batches": 25},
        "extended": {"iters": 1500, "batch_size": 2, "learning_rate": 5e-6, "val_batches": 50},
    }
    if preset not in presets:
        raise ValueError("preset must be smoke, standard, or extended")
    with transaction() as con:
        dataset = con.execute("SELECT artifact_path FROM datasets WHERE id=?", (dataset_id,)).fetchone()
        if not dataset:
            raise KeyError(dataset_id)
        active = con.execute("SELECT 1 FROM experiments WHERE status IN ('queued','running')").fetchone()
        if active:
            raise RuntimeError("A training job is already queued or running")
    experiment_id = new_id("exp")
    output = path_for("models", experiment_id)
    output.mkdir(parents=True, exist_ok=False)
    config = {
        "model": model_id or settings.model_id,
        "dataset_id": dataset_id,
        "preset": preset,
        "fine_tune_type": "lora",
        "seed": 42,
        "num_layers": 16,
        "rank": 8,
        "scale": 16.0,
        "dropout": 0.0,
        "max_seq_length": 2048,
        "target_layers": ["self_attn.q_proj", "self_attn.v_proj"],
        **presets[preset],
    }
    manifest: dict[str, Any] = {
        "experiment_id": experiment_id,
        "config": config,
        "prompt_version": ProductExtractionTask.prompt_version,
        "mlx": mlx,
        "adapter_path": str(output),
        "started_at": now(),
    }
    (output / "training-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with transaction() as con:
        con.execute(
            "INSERT INTO experiments VALUES(?,?,?,?,?,?,?)",
            (
                experiment_id,
                dataset_id,
                "queued",
                json.dumps(config),
                json.dumps(manifest),
                manifest["started_at"],
                manifest["started_at"],
            ),
        )
    log = output / "training.log"
    command = [
        sys.executable,
        "-m",
        "llm_feedback_finetuning_pipeline.train_worker",
        experiment_id,
        str(dataset["artifact_path"]),
        str(output),
        json.dumps(config),
    ]
    with log.open("w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT, start_new_session=True)
    manifest.update({"pid": proc.pid, "log": str(log), "status": "running"})
    with transaction() as con:
        con.execute(
            "UPDATE experiments SET status=?,manifest=?,updated_at=? WHERE id=? AND status='queued'",
            ("running", json.dumps(manifest), now(), experiment_id),
        )
    return get_job(experiment_id)


def get_job(experiment_id: str) -> dict[str, Any]:
    with transaction() as con:
        row = con.execute("SELECT * FROM experiments WHERE id=?", (experiment_id,)).fetchone()
    if row is None:
        raise KeyError(experiment_id)
    return {**dict(row), "config": json.loads(row["config"]), "manifest": json.loads(row["manifest"] or "{}")}


def list_jobs() -> list[dict[str, Any]]:
    with transaction() as con:
        rows = con.execute("SELECT * FROM experiments ORDER BY created_at DESC").fetchall()
    return [
        {**dict(row), "config": json.loads(row["config"]), "manifest": json.loads(row["manifest"] or "{}")}
        for row in rows
    ]


def cancel_job(experiment_id: str) -> dict[str, Any]:
    job = get_job(experiment_id)
    if job["status"] != "running":
        raise ValueError(f"job is {job['status']}; only running jobs can be cancelled")
    pid = int(job["manifest"].get("pid", 0))
    if not pid:
        raise ValueError("job has no tracked process id")
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    with transaction() as con:
        con.execute("UPDATE experiments SET status='cancelled',updated_at=? WHERE id=?", (now(), experiment_id))
    return {"id": experiment_id, "status": "cancelled"}


def reconcile_jobs() -> None:
    """Mark child jobs interrupted after restart; never restart training implicitly."""
    import psutil

    for job in list_jobs():
        if job["status"] != "running":
            continue
        pid = job["manifest"].get("pid")
        try:
            command = " ".join(psutil.Process(int(pid)).cmdline())
            alive = "llm_feedback_finetuning_pipeline.train_worker" in command and job["id"] in command
        except (psutil.Error, TypeError, ValueError):
            alive = False
        if not alive:
            with transaction() as con:
                manifest = job["manifest"]
                manifest.update(
                    {"status": "interrupted", "error": "Training worker stopped; inspect log and explicitly restart."}
                )
                con.execute(
                    "UPDATE experiments SET status='interrupted',manifest=?,updated_at=? WHERE id=?",
                    (json.dumps(manifest), now(), job["id"]),
                )
