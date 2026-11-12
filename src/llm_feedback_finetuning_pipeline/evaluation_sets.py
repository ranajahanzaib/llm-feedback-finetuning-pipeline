from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .db import new_id, path_for, transaction
from .feedback import preview
from .provenance import input_fingerprint
from .tasks.products import ProductExtractionTask


def create_evaluation_set(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Freeze independent labeled examples outside the feedback/training workflow."""
    report = preview(records)
    if report["invalid"]:
        raise ValueError(json.dumps({"invalid": report["invalid"]}))
    if report["duplicates"] or report["conflicts"] or report["near_duplicates"]:
        raise ValueError("Evaluation-set inputs must be unique; resolve duplicate or near-duplicate records first")
    if report["privacy_warnings"]:
        raise ValueError("Redact privacy-flagged evaluation records before freezing the set")
    if report["quality_warnings"]:
        raise ValueError("Remove placeholder labels before freezing the evaluation set")
    examples = report["records"]
    if len(examples) < 3:
        raise ValueError("An independent evaluation set requires at least 3 records")
    input_hashes = [input_fingerprint(item["input_text"]) for item in examples]
    input_hash_set = set(input_hashes)
    with transaction() as con:
        feedback_rows = con.execute("SELECT id,input_text,source_group_id FROM feedback").fetchall()
        datasets = con.execute("SELECT manifest FROM datasets").fetchall()
        previous_sets = con.execute("SELECT manifest FROM evaluation_sets").fetchall()
    overlap_feedback = [row["id"] for row in feedback_rows if input_fingerprint(row["input_text"]) in input_hash_set]
    if overlap_feedback:
        raise ValueError("Evaluation inputs overlap existing feedback; use a separately sourced holdout")
    training_hashes = {
        fingerprint for row in datasets for fingerprint in json.loads(row["manifest"]).get("input_hashes", [])
    }
    if input_hash_set & training_hashes:
        raise ValueError("Evaluation inputs overlap an existing training dataset")
    prior_eval_hashes = {
        fingerprint for row in previous_sets for fingerprint in json.loads(row["manifest"]).get("input_hashes", [])
    }
    if input_hash_set & prior_eval_hashes:
        raise ValueError("Evaluation inputs overlap another frozen evaluation set")

    evaluation_set_id = new_id("evalset")
    root = path_for("evaluation_sets", evaluation_set_id)
    root.mkdir(parents=True, exist_ok=False)
    rows = [
        json.dumps(
            ProductExtractionTask.serialize_example(item["input_text"], item["corrected_output"]),
            ensure_ascii=False,
        )
        for item in examples
    ]
    content = "\n".join(rows) + "\n"
    (root / "test.jsonl").write_text(content, encoding="utf-8")
    manifest = {
        "evaluation_set_id": evaluation_set_id,
        "task_id": ProductExtractionTask.task_id,
        "schema_version": ProductExtractionTask.schema_version,
        "prompt_version": ProductExtractionTask.prompt_version,
        "count": len(examples),
        "source_record_ids": [item["source_record_id"] for item in examples if item["source_record_id"]],
        "input_hashes": input_hashes,
        "file_sha256": {"test.jsonl": hashlib.sha256(content.encode()).hexdigest()},
        "created_at": datetime.now(UTC).isoformat(),
        "eligibility": "separate labeled holdout; not stored as feedback and never exported to training datasets",
    }
    manifest_text = json.dumps(manifest, indent=2, sort_keys=True)
    (root / "manifest.json").write_text(manifest_text + "\n", encoding="utf-8")
    with transaction() as con:
        con.execute(
            "INSERT INTO evaluation_sets VALUES(?,?,?,?,?)",
            (evaluation_set_id, ProductExtractionTask.task_id, manifest_text, str(root), manifest["created_at"]),
        )
    return manifest


def list_evaluation_sets() -> list[dict[str, Any]]:
    with transaction() as con:
        rows = con.execute(
            "SELECT id,manifest,artifact_path,created_at FROM evaluation_sets ORDER BY created_at DESC"
        ).fetchall()
    return [
        {
            "id": row["id"],
            "manifest": json.loads(row["manifest"]),
            "artifact_path": row["artifact_path"],
            "created_at": row["created_at"],
        }
        for row in rows
    ]


def get_evaluation_set(evaluation_set_id: str) -> dict[str, Any]:
    with transaction() as con:
        row = con.execute(
            "SELECT id,manifest,artifact_path,created_at FROM evaluation_sets WHERE id=?", (evaluation_set_id,)
        ).fetchone()
    if row is None:
        raise KeyError(evaluation_set_id)
    return {
        "id": row["id"],
        "manifest": json.loads(row["manifest"]),
        "artifact_path": row["artifact_path"],
        "created_at": row["created_at"],
    }


def verify_evaluation_set(evaluation_set_id: str) -> dict[str, Any]:
    evaluation_set = get_evaluation_set(evaluation_set_id)
    root = Path(evaluation_set["artifact_path"]).resolve()
    if root != path_for("evaluation_sets", evaluation_set_id).resolve():
        raise ValueError("Evaluation-set path does not match its managed artifact location")
    mismatches = []
    for filename, expected in evaluation_set["manifest"].get("file_sha256", {}).items():
        path = root / filename
        if path.is_symlink() or not path.resolve().is_relative_to(root) or not path.is_file():
            mismatches.append(filename)
        elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            mismatches.append(filename)
    if mismatches:
        raise ValueError("Evaluation-set integrity check failed for: " + ", ".join(mismatches))
    return {"evaluation_set_id": evaluation_set_id, "verified_files": ["test.jsonl"]}
