from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .db import new_id, path_for, transaction
from .provenance import input_fingerprint
from .tasks.products import ProductExtractionTask


def build_dataset(
    *, seed: int = 42, train_pct: float = 0.8, valid_pct: float = 0.1, test_pct: float = 0.1
) -> dict[str, Any]:
    if any(p < 0 for p in (train_pct, valid_pct, test_pct)) or abs(train_pct + valid_pct + test_pct - 1) > 1e-8:
        raise ValueError("train_pct, valid_pct and test_pct must be non-negative and sum to 1")
    with transaction() as con:
        all_rows = con.execute(
            "SELECT * FROM feedback WHERE status='approved' AND conflict=0 AND duplicate_of IS NULL "
            "AND privacy_flags='[]' ORDER BY id"
        ).fetchall()
        prior_datasets = con.execute("SELECT id,manifest,artifact_path FROM datasets").fetchall()
        frozen_evaluation_sets = con.execute("SELECT manifest FROM evaluation_sets").fetchall()
    frozen_hashes = {
        fingerprint
        for row in frozen_evaluation_sets
        for fingerprint in json.loads(row["manifest"]).get("input_hashes", [])
    }
    excluded_holdout_ids = [row["id"] for row in all_rows if input_fingerprint(row["input_text"]) in frozen_hashes]
    excluded_holdout_set = set(excluded_holdout_ids)
    rows = [row for row in all_rows if row["id"] not in excluded_holdout_set]
    if len(rows) < 3:
        raise ValueError(f"Need at least 3 approved unique records; found {len(rows)}")
    grouped: dict[str, list[Any]] = {}
    for row in rows:
        group = row["source_group_id"] or row["id"]
        grouped.setdefault(group, []).append(row)
    groups = list(grouped)
    if test_pct > 0 and len(groups) < 3:
        raise ValueError("At least three independent source groups are required for train/validation/test separation")
    if valid_pct > 0 and len(groups) < 2:
        raise ValueError("At least two independent source groups are required for validation")
    random.Random(seed).shuffle(groups)
    target = {
        "train": train_pct * len(rows),
        "valid": valid_pct * len(rows),
        "test": test_pct * len(rows),
    }
    splits: dict[str, list[Any]] = {name: [] for name in target}
    for group in groups:
        bucket = min(target, key=lambda name: len(splits[name]) / max(target[name], 0.001))
        splits[bucket].extend(grouped[group])
    # Make evaluation splits non-empty when there are enough independent groups.
    if len(groups) >= 3:
        for bucket in ("valid", "test"):
            if target[bucket] and not splits[bucket]:
                donor = max(
                    (name for name in target if len(splits[name]) > 1),
                    key=lambda name: len(splits[name]),
                )
                splits[bucket].append(splits[donor].pop())
    id_ = new_id("ds")
    dest = path_for("datasets", id_)
    dest.mkdir(parents=True, exist_ok=False)
    hashes: dict[str, str] = {}
    counts: dict[str, int] = {}
    labels: Counter[str] = Counter()
    for bucket, examples in splits.items():
        lines = []
        for row in examples:
            output = ProductExtractionTask.normalize(json.loads(row["corrected_output"]))
            for key, value in output.items():
                labels[f"{key}:{value if value is not None else '<null>'}"] += 1
            lines.append(
                json.dumps(
                    ProductExtractionTask.serialize_example(row["input_text"], output),
                    ensure_ascii=False,
                )
            )
        content = "\n".join(lines) + ("\n" if lines else "")
        (dest / f"{bucket}.jsonl").write_text(content, encoding="utf-8")
        hashes[f"{bucket}.jsonl"] = hashlib.sha256(content.encode()).hexdigest()
        counts[bucket] = len(examples)
    input_hashes = [input_fingerprint(row["input_text"]) for row in rows]
    prior_test_hashes: set[str] = set()
    for prior in prior_datasets:
        prior_manifest = json.loads(prior["manifest"])
        prior_test_hashes.update(prior_manifest.get("test_input_hashes", []))
        # Backward-compatible contamination scan for snapshots written before test hashes were recorded.
        test_file = path_for("datasets", prior["id"]) / "test.jsonl"
        if test_file.is_file():
            for line in test_file.read_text(encoding="utf-8").splitlines():
                if line:
                    example = json.loads(line)
                    text = example["messages"][1]["content"]
                    prior_test_hashes.add(input_fingerprint(text))
    current_hashes = {input_fingerprint(row["input_text"]) for row in rows}
    contamination_hashes = sorted(current_hashes & prior_test_hashes)
    output_nulls: Counter[str] = Counter()
    for row in rows:
        for key, value in ProductExtractionTask.normalize(json.loads(row["corrected_output"])).items():
            if value is None:
                output_nulls[key] += 1
    split_feedback_ids = {name: [row["id"] for row in values] for name, values in splits.items()}
    test_input_hashes = [input_fingerprint(row["input_text"]) for row in splits["test"]]
    manifest = {
        "dataset_id": id_,
        "task_id": ProductExtractionTask.task_id,
        "schema_version": ProductExtractionTask.schema_version,
        "prompt_version": ProductExtractionTask.prompt_version,
        "selected_feedback_ids": [r["id"] for r in rows],
        "excluded_frozen_evaluation_feedback_ids": excluded_holdout_ids,
        "split_feedback_ids": split_feedback_ids,
        "source_counts": dict(Counter(r["source"] for r in rows)),
        "eligibility": "approved, conflict-free, unique inputs only; imported records default to pending",
        "normalization": "casefold category/color/material/gender; uppercase alpha sizes; product-extraction 1.0.0",
        "deduplication": "case-insensitive exact input deduplication; near duplicates not automatically merged",
        "split": {
            "seed": seed,
            "train": train_pct,
            "valid": valid_pct,
            "test": test_pct,
            "strategy": "deterministic shuffled source-group assignment",
        },
        "counts": counts,
        "statistics": {
            "approved_examples": len(rows),
            "average_input_characters": round(sum(len(r["input_text"]) for r in rows) / len(rows), 2),
            "missing_attribute_counts": dict(output_nulls),
            "source_coverage": dict(Counter(r["source"] for r in rows)),
            "category_coverage": dict(
                Counter(
                    str(ProductExtractionTask.normalize(json.loads(r["corrected_output"]))["category"]) for r in rows
                )
            ),
        },
        "label_distribution": dict(sorted(labels.items())),
        "input_hashes": input_hashes,
        "test_input_hashes": test_input_hashes,
        "file_sha256": hashes,
        "created_at": datetime.now(UTC).isoformat(),
        "warnings": (
            (
                ["Small synthetic or business datasets do not support robust generalization claims."]
                if len(rows) < 30
                else []
            )
            + (
                [
                    f"{len(contamination_hashes)} input(s) overlap an earlier frozen test set; "
                    "later results are contaminated."
                ]
                if contamination_hashes
                else []
            )
            + (
                [
                    f"Excluded {len(excluded_holdout_ids)} approved feedback record(s) matching "
                    "an independent frozen evaluation set."
                ]
                if excluded_holdout_ids
                else []
            )
        ),
        "contamination": {
            "previous_frozen_test_overlap_count": len(contamination_hashes),
            "overlap_input_hashes": contamination_hashes,
        },
    }
    manifest_text = json.dumps(manifest, indent=2, sort_keys=True)
    (dest / "manifest.json").write_text(manifest_text + "\n", encoding="utf-8")
    with transaction() as con:
        con.execute(
            "INSERT INTO datasets VALUES(?,?,?,?,?)",
            (id_, ProductExtractionTask.task_id, manifest_text, str(dest), manifest["created_at"]),
        )
    return manifest


def list_datasets() -> list[dict[str, Any]]:
    with transaction() as con:
        rows = con.execute(
            "SELECT id,manifest,artifact_path,created_at FROM datasets ORDER BY created_at DESC"
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


def get_dataset(dataset_id: str) -> dict[str, Any]:
    with transaction() as con:
        row = con.execute(
            "SELECT id,manifest,artifact_path,created_at FROM datasets WHERE id=?", (dataset_id,)
        ).fetchone()
    if row is None:
        raise KeyError(dataset_id)
    return {
        "id": row["id"],
        "manifest": json.loads(row["manifest"]),
        "artifact_path": row["artifact_path"],
        "created_at": row["created_at"],
    }


def verify_dataset_artifacts(dataset_id: str) -> dict[str, Any]:
    dataset = get_dataset(dataset_id)
    manifest = dataset["manifest"]
    root = Path(dataset["artifact_path"]).resolve()
    expected_root = path_for("datasets", dataset_id).resolve()
    if root != expected_root:
        raise ValueError("Dataset artifact path does not match its managed artifact location")
    mismatches = []
    for filename, expected in manifest.get("file_sha256", {}).items():
        path = root / filename
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            mismatches.append(filename)
        elif not path.is_file():
            mismatches.append(filename)
        elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            mismatches.append(filename)
    if mismatches:
        raise ValueError("Dataset snapshot integrity check failed for: " + ", ".join(mismatches))
    return {"dataset_id": dataset_id, "verified_files": sorted(manifest.get("file_sha256", {}))}


def diff_datasets(old_id: str, new_id: str) -> dict[str, Any]:
    old, new = get_dataset(old_id), get_dataset(new_id)
    a, b = old["manifest"], new["manifest"]
    old_ids, new_ids = set(a.get("selected_feedback_ids", [])), set(b.get("selected_feedback_ids", []))
    return {
        "from_dataset": old_id,
        "to_dataset": new_id,
        "record_count_delta": len(new_ids) - len(old_ids),
        "added_feedback_ids": sorted(new_ids - old_ids),
        "removed_feedback_ids": sorted(old_ids - new_ids),
        "schema_changed": a.get("schema_version") != b.get("schema_version"),
        "label_distribution_delta": {
            key: b.get("label_distribution", {}).get(key, 0) - a.get("label_distribution", {}).get(key, 0)
            for key in sorted(set(a.get("label_distribution", {})) | set(b.get("label_distribution", {})))
        },
        "quality_warnings": b.get("warnings", []),
        "contamination": b.get("contamination", {}),
    }
