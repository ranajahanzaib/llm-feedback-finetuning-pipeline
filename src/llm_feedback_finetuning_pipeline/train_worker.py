"""Controlled child process for a single local MLX-LM LoRA training run."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import re
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .db import now, transaction


def run(experiment_id: str, data_path: Path, output: Path, config: dict[str, Any]) -> int:
    from .registry import adapter_checksum
    from .training import cached_model_revision

    started = time.perf_counter()
    mlx_config = {
        "model": config["model"],
        "data": str(data_path),
        "train": True,
        "fine_tune_type": "lora",
        "adapter_path": str(output),
        "iters": config["iters"],
        "batch_size": config["batch_size"],
        "learning_rate": config["learning_rate"],
        "num_layers": config["num_layers"],
        "val_batches": config["val_batches"],
        "max_seq_length": config["max_seq_length"],
        "mask_prompt": True,
        "seed": config["seed"],
        "lora_parameters": {
            "keys": config["target_layers"],
            "rank": config["rank"],
            "scale": config["scale"],
            "dropout": config["dropout"],
        },
    }
    config_path = output / "mlx-lm-config.yaml"
    config_path.write_text(json.dumps(mlx_config, indent=2) + "\n", encoding="utf-8")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "mlx_lm", "lora", "--config", str(config_path)],
            cwd=output,
            timeout=12 * 60 * 60,
            check=False,
        )
        adapter = any(output.glob("*.safetensors"))
        status = "completed" if result.returncode == 0 and adapter else "failed"
        detail: dict[str, Any] = {"returncode": result.returncode, "adapter_present": adapter}
    except subprocess.TimeoutExpired:
        status = "failed"
        detail = {"error": "Training exceeded the 12 hour safety timeout"}
    except Exception as exc:
        status = "failed"
        detail = {"error": f"{type(exc).__name__}: {exc}"}
    with transaction() as con:
        row = con.execute("SELECT manifest FROM experiments WHERE id=?", (experiment_id,)).fetchone()
        manifest = json.loads(row[0]) if row else {}
        log_path = output / "training.log"
        log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
        train_losses = re.findall(r"Train loss\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", log_text, re.I)
        valid_losses = re.findall(r"(?:Val|Validation) loss\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", log_text, re.I)
        iterations = re.findall(r"(?:Iter|Iteration)\s+(\d+)", log_text, re.I)
        artifacts = sorted(output.glob("*.safetensors"))
        adapter_hash = adapter_checksum(output) if artifacts else None
        source_digest = hashlib.sha256()
        for source in sorted(Path(__file__).parent.rglob("*.py")):
            source_digest.update(source.read_bytes())
        manifest.update(
            {
                **detail,
                "finished_at": now(),
                "status": status,
                "training_config": str(config_path),
                "training_duration_seconds": round(time.perf_counter() - started, 3),
                "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "adapter_sha256": adapter_hash,
                "adapter_artifact_bytes": sum(item.stat().st_size for item in artifacts),
                "dataset_manifest_sha256": hashlib.sha256((data_path / "manifest.json").read_bytes()).hexdigest(),
                "source_tree_sha256": source_digest.hexdigest(),
                "python_version": platform.python_version(),
                "mlx_version": importlib.metadata.version("mlx"),
                "mlx_lm_version": importlib.metadata.version("mlx-lm"),
                "resolved_base_model_revision": cached_model_revision(config["model"]),
                "training_metrics": {
                    "final_train_loss": float(train_losses[-1]) if train_losses else None,
                    "final_validation_loss": float(valid_losses[-1]) if valid_losses else None,
                    "last_logged_iteration": int(iterations[-1]) if iterations else None,
                    "source": str(log_path),
                },
            }
        )
        con.execute(
            "UPDATE experiments SET status=?,manifest=?,updated_at=? WHERE id=?",
            (status, json.dumps(manifest), now(), experiment_id),
        )
    if status == "completed":
        from .registry import register_model

        register_model(
            config["model"],
            adapter_path=output.name,
            dataset_id=config["dataset_id"],
            experiment_id=experiment_id,
            metrics={
                "adapter_sha256": manifest["adapter_sha256"],
                "adapter_artifact_bytes": manifest["adapter_artifact_bytes"],
                "resolved_base_model_revision": manifest["resolved_base_model_revision"],
                **manifest["training_metrics"],
                "n": 0,
            },
        )
    return 0 if status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(run(sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), json.loads(sys.argv[4])))
