from __future__ import annotations

import csv
import difflib
import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .db import asdict, new_id, now, transaction
from .privacy import flags, redact
from .tasks.products import ProductAttributes, ProductExtractionTask


def validate_record(raw: dict[str, Any], line: int) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"record {line}: each imported record must be a JSON object")
    text = str(raw.get("input_text", "")).strip()
    if not text:
        raise ValueError(f"record {line}: input_text is required")
    ProductExtractionTask.validate_input(text)
    output = raw.get("corrected_output")
    if isinstance(output, str):
        output = json.loads(output)
    if not isinstance(output, dict):
        raise ValueError(f"record {line}: corrected_output must be an object or JSON object")
    ProductAttributes.model_validate(output)
    normalized_output = ProductExtractionTask.normalize(output)
    suspicious_labels = [
        key
        for key, value in normalized_output.items()
        if isinstance(value, str) and value.casefold() in {"test", "todo", "tbd", "placeholder", "asdf"}
    ]
    return {
        "id": new_id("fb"),
        "task_id": ProductExtractionTask.task_id,
        "schema_version": ProductExtractionTask.schema_version,
        "input_text": text,
        "original_prediction": raw.get("original_prediction"),
        "corrected_output": normalized_output,
        "source": str(raw.get("source") or "import"),
        "source_record_id": raw.get("source_record_id"),
        "source_group_id": raw.get("source_group_id"),
        "original_model_id": raw.get("original_model_id"),
        "prompt_version": raw.get("prompt_version"),
        "model_generation_metadata": raw.get("model_generation_metadata"),
        "status": "pending",
        "notes": raw.get("notes"),
        "privacy_flags": sorted(
            set(
                flags(text)
                + flags(json.dumps(output))
                + flags(json.dumps(raw.get("original_prediction")))
                + flags(json.dumps(raw.get("model_generation_metadata")))
                + flags(json.dumps(raw.get("notes")))
                + flags(str(raw.get("source_record_id") or ""))
                + flags(str(raw.get("source_group_id") or ""))
            )
        ),
        "quality_warnings": [f"placeholder_label:{field}" for field in suspicious_labels],
        "created_at": now(),
        "duplicate_of": None,
        "conflict": False,
    }


def read_import(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError("Import source must be an existing file")
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as file:
            return list(csv.DictReader(file))
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        records = []
        for line, content in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if content.strip():
                try:
                    records.append(json.loads(content))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"line {line}: invalid JSON: {exc.msg}") from exc
        return records
    raise ValueError("Supported imports are CSV and JSONL")


def preview(records: list[dict[str, Any]]) -> dict[str, Any]:
    valid, invalid = [], []
    for line, raw in enumerate(records, 1):
        try:
            valid.append(validate_record(raw, line))
        except (ValueError, ValidationError, TypeError, json.JSONDecodeError) as exc:
            invalid.append({"record": line, "error": str(exc)})
    seen: dict[str, dict[str, Any]] = {}
    duplicates, conflicts = [], []
    for item in valid:
        prior = seen.get(item["input_text"].casefold())
        if prior:
            if prior["corrected_output"] == item["corrected_output"]:
                item["duplicate_of"] = prior["id"]
                duplicates.append(item["id"])
            else:
                item["conflict"] = prior["conflict"] = True
                item["status"] = prior["status"] = "quarantined"
                conflicts.append([prior["id"], item["id"]])
        else:
            seen[item["input_text"].casefold()] = item
    near_duplicates = []
    if len(valid) <= 5000:
        unique = [item for item in valid if not item["duplicate_of"]]
        token_index: dict[str, list[int]] = {}
        normalized = [" ".join(item["input_text"].casefold().split()) for item in unique]
        for index, text in enumerate(normalized):
            tokens = set(text.split())
            candidates: set[int] = set()
            for token in sorted(tokens):
                candidates.update(token_index.get(token, [])[-100:])
            for prior_index in sorted(candidates):
                prior_tokens = set(normalized[prior_index].split())
                jaccard = len(tokens & prior_tokens) / max(1, len(tokens | prior_tokens))
                if jaccard >= 0.7 and 0.75 <= difflib.SequenceMatcher(None, text, normalized[prior_index]).ratio() < 1:
                    near_duplicates.append([unique[prior_index]["id"], unique[index]["id"]])
            for token in sorted(tokens):
                token_index.setdefault(token, []).append(index)
    return {
        "valid_count": len(valid),
        "invalid": invalid,
        "duplicates": duplicates,
        "conflicts": conflicts,
        "near_duplicates": near_duplicates,
        "quality_warnings": [
            {"id": item["id"], "warnings": item["quality_warnings"]} for item in valid if item["quality_warnings"]
        ],
        "privacy_warnings": [
            {"id": item["id"], "flags": item["privacy_flags"]} for item in valid if item["privacy_flags"]
        ],
        "records": valid,
    }


def import_records(records: list[dict[str, Any]], *, commit: bool = False) -> dict[str, Any]:
    result = preview(records)
    record_ids = [item["id"] for item in result["records"]]
    database_conflicts: list[list[str]] = []
    if commit and not result["invalid"]:
        with transaction() as con:
            existing = {
                r["input_text"].casefold(): r
                for r in con.execute("SELECT id,input_text,corrected_output,status,privacy_flags FROM feedback")
            }
            for item in result["records"]:
                prior = existing.get(item["input_text"].casefold())
                if prior is not None:
                    if json.loads(prior["corrected_output"]) != item["corrected_output"]:
                        item["conflict"] = True
                        item["status"] = "quarantined"
                        when = now()
                        con.execute(
                            "INSERT INTO review_history VALUES(?,?,?,?,?,?,?,?)",
                            (
                                new_id("rv"),
                                prior["id"],
                                prior["status"],
                                "quarantined",
                                "system:conflict-detection",
                                f"Conflicting correction {item['id']} imported; adjudication required.",
                                when,
                                prior["corrected_output"]
                                if not json.loads(prior["privacy_flags"])
                                else '{"redacted":true}',
                            ),
                        )
                        con.execute(
                            "UPDATE feedback SET status='quarantined',conflict=1,reviewer=?,reviewed_at=?,"
                            "notes=COALESCE(notes,'') || ? WHERE id=?",
                            (
                                "system:conflict-detection",
                                when,
                                f"\nConflicting correction {item['id']} imported; adjudication required.",
                                prior["id"],
                            ),
                        )
                        database_conflicts.append([prior["id"], item["id"]])
                    else:
                        item["duplicate_of"] = "existing"
                con.execute(
                    """INSERT INTO feedback(id,task_id,schema_version,input_text,original_prediction,
                    corrected_output,source,source_record_id,source_group_id,original_model_id,prompt_version,
                    model_generation_metadata,status,notes,privacy_flags,created_at,duplicate_of,conflict)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        item["id"],
                        item["task_id"],
                        item["schema_version"],
                        item["input_text"],
                        json.dumps(item["original_prediction"]) if item["original_prediction"] is not None else None,
                        json.dumps(item["corrected_output"]),
                        item["source"],
                        item["source_record_id"],
                        item["source_group_id"],
                        item["original_model_id"],
                        item["prompt_version"],
                        json.dumps(item["model_generation_metadata"])
                        if item["model_generation_metadata"] is not None
                        else None,
                        item["status"],
                        item["notes"],
                        json.dumps(item["privacy_flags"]),
                        item["created_at"],
                        item["duplicate_of"],
                        int(item["conflict"]),
                    ),
                )
        result["committed"] = True
    else:
        result["committed"] = False
    result["feedback_ids"] = record_ids
    result["database_conflicts"] = database_conflicts
    result.pop("records", None)
    return result


def list_feedback(
    status: str | None = None,
    *,
    source: str | None = None,
    task_id: str | None = None,
    created_after: str | None = None,
    created_before: str | None = None,
) -> list[dict[str, Any]]:
    clauses, params = [], []
    for column, value in (("status", status), ("source", source), ("task_id", task_id)):
        if value:
            clauses.append(f"{column}=?")
            params.append(value)
    for operator, value in ((">=", created_after), ("<=", created_before)):
        if value:
            clauses.append(f"created_at{operator}?")
            params.append(value)
    with transaction() as con:
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        rows = con.execute(f"SELECT * FROM feedback{where} ORDER BY created_at DESC", params)
        return [asdict(row) or {} for row in rows]


def get_feedback(feedback_id: str) -> dict[str, Any]:
    with transaction() as con:
        row = con.execute("SELECT * FROM feedback WHERE id=?", (feedback_id,)).fetchone()
        if row is None:
            raise KeyError(feedback_id)
        current = asdict(row) or {}
        history_rows = con.execute(
            "SELECT * FROM review_history WHERE feedback_id=? ORDER BY changed_at", (feedback_id,)
        ).fetchall()
    history = [dict(row) for row in history_rows]
    for index, entry in enumerate(history):
        before = json.loads(entry["prior_output"])
        after = (
            json.loads(history[index + 1]["prior_output"]) if index + 1 < len(history) else current["corrected_output"]
        )
        entry["field_changes"] = (
            {}
            if before == {"redacted": True}
            else {
                field: {"before": before.get(field), "after": after.get(field)}
                for field in sorted(set(before) | set(after))
                if before.get(field) != after.get(field)
            }
        )
    current["history"] = history
    return current


def review(feedback_id: str, decision: str, reviewer: str, notes: str | None = None) -> dict[str, Any]:
    if decision not in {"approved", "rejected", "quarantined"}:
        raise ValueError("decision must be approved, rejected, or quarantined")
    notes = redact(notes) if notes else None
    with transaction() as con:
        row = con.execute("SELECT * FROM feedback WHERE id=?", (feedback_id,)).fetchone()
        if row is None:
            raise KeyError(feedback_id)
        if row["status"] != "pending":
            raise ValueError(f"cannot review a {row['status']} record again; create a new correction")
        if decision == "approved" and json.loads(row["privacy_flags"]):
            raise ValueError("redact or exclude privacy-flagged feedback before approval")
        when = now()
        con.execute(
            "INSERT INTO review_history VALUES(?,?,?,?,?,?,?,?)",
            (
                new_id("rv"),
                feedback_id,
                row["status"],
                decision,
                reviewer,
                notes,
                when,
                row["corrected_output"] if not json.loads(row["privacy_flags"]) else '{"redacted":true}',
            ),
        )
        con.execute(
            "UPDATE feedback SET status=?,reviewer=?,notes=COALESCE(?,notes),reviewed_at=? WHERE id=?",
            (decision, reviewer, notes, when, feedback_id),
        )
        return asdict(con.execute("SELECT * FROM feedback WHERE id=?", (feedback_id,)).fetchone()) or {}


def resolve_conflict(feedback_id: str, reviewer: str, reason: str) -> dict[str, Any]:
    """Explicitly select one quarantined correction for a conflicting input."""
    if not reason.strip():
        raise ValueError("a non-empty adjudication reason is required")
    reason = redact(reason)
    with transaction() as con:
        row = con.execute("SELECT * FROM feedback WHERE id=?", (feedback_id,)).fetchone()
        if row is None:
            raise KeyError(feedback_id)
        if row["status"] != "quarantined" or not row["conflict"]:
            raise ValueError("record is not a quarantined conflict")
        if json.loads(row["privacy_flags"]):
            raise ValueError("redact privacy-flagged feedback before conflict adjudication")
        try:
            ProductAttributes.model_validate(json.loads(row["corrected_output"]))
        except (ValidationError, json.JSONDecodeError) as exc:
            raise ValueError("selected correction no longer passes task schema validation") from exc
        when = now()
        con.execute(
            "INSERT INTO review_history VALUES(?,?,?,?,?,?,?,?)",
            (
                new_id("rv"),
                feedback_id,
                row["status"],
                "approved",
                reviewer,
                "Conflict adjudication: " + reason.strip(),
                when,
                row["corrected_output"],
            ),
        )
        con.execute(
            "UPDATE feedback SET status='approved',conflict=0,reviewer=?,reviewed_at=?,notes=COALESCE(notes,'') || ? "
            "WHERE id=?",
            (reviewer, when, "\nConflict adjudication: " + reason.strip(), feedback_id),
        )
        return asdict(con.execute("SELECT * FROM feedback WHERE id=?", (feedback_id,)).fetchone()) or {}


def redact_feedback(feedback_id: str, reviewer: str) -> dict[str, Any]:
    """Redact recognized patterns from a pending input without copying raw text to history."""
    with transaction() as con:
        row = con.execute("SELECT * FROM feedback WHERE id=?", (feedback_id,)).fetchone()
        if row is None:
            raise KeyError(feedback_id)
        if row["status"] not in {"pending", "quarantined"}:
            raise ValueError("only pending or quarantined feedback can be redacted")
        input_text = redact(row["input_text"])
        corrected = json.loads(row["corrected_output"])
        original = json.loads(row["original_prediction"]) if row["original_prediction"] else None
        metadata = json.loads(row["model_generation_metadata"]) if row["model_generation_metadata"] else None
        source_record_id = row["source_record_id"]
        source_group_id = row["source_group_id"]

        def clean(value: Any) -> Any:
            if isinstance(value, str):
                return redact(value)
            if isinstance(value, dict):
                return {redact(str(key)): clean(child) for key, child in value.items()}
            if isinstance(value, list):
                return [clean(child) for child in value]
            return value

        clean_corrected, clean_original, clean_metadata = clean(corrected), clean(original), clean(metadata)
        clean_notes = redact(row["notes"] or "")
        clean_source_record_id = redact(source_record_id) if source_record_id else None
        clean_source_group_id = redact(source_group_id) if source_group_id else None
        if (
            input_text,
            clean_corrected,
            clean_original,
            clean_metadata,
            clean_notes,
            clean_source_record_id,
            clean_source_group_id,
        ) == (
            row["input_text"],
            corrected,
            original,
            metadata,
            row["notes"] or "",
            source_record_id,
            source_group_id,
        ):
            raise ValueError("no recognized patterns were found to redact")
        old_flags = json.loads(row["privacy_flags"])
        when = now()
        note = "Redacted detected input patterns: " + ", ".join(old_flags)
        con.execute(
            "INSERT INTO review_history VALUES(?,?,?,?,?,?,?,?)",
            (new_id("rv"), feedback_id, row["status"], row["status"], reviewer, note, when, '{"redacted":true}'),
        )
        con.execute(
            "UPDATE feedback SET input_text=?,corrected_output=?,original_prediction=?,"
            "model_generation_metadata=?,source_record_id=?,source_group_id=?,privacy_flags='[]',notes=? || ? "
            "WHERE id=?",
            (
                input_text,
                json.dumps(clean_corrected),
                json.dumps(clean_original) if clean_original is not None else None,
                json.dumps(clean_metadata) if clean_metadata is not None else None,
                clean_source_record_id,
                clean_source_group_id,
                clean_notes,
                "\n" + note,
                feedback_id,
            ),
        )
        return asdict(con.execute("SELECT * FROM feedback WHERE id=?", (feedback_id,)).fetchone()) or {}


def delete_feedback(feedback_id: str, operator: str, reason: str) -> dict[str, Any]:
    """Delete a source record only when no immutable dataset references it."""
    if not operator.strip() or not reason.strip():
        raise ValueError("operator and deletion reason are required")
    reason = redact(reason.strip())[:2000]
    with transaction() as con:
        row = con.execute("SELECT * FROM feedback WHERE id=?", (feedback_id,)).fetchone()
        if row is None:
            raise KeyError(feedback_id)
        for dataset_row in con.execute("SELECT manifest FROM datasets"):
            manifest = json.loads(dataset_row["manifest"])
            if feedback_id in manifest.get("selected_feedback_ids", []):
                raise ValueError(
                    "feedback is referenced by an immutable dataset; delete or rebuild artifacts separately"
                )
        con.execute("DELETE FROM review_history WHERE feedback_id=?", (feedback_id,))
        con.execute("DELETE FROM feedback WHERE id=?", (feedback_id,))
        deleted_at = now()
        audit_id = new_id("del")
        con.execute(
            "INSERT INTO deletion_audit VALUES(?,?,?,?,?,?)",
            (audit_id, feedback_id, row["status"], operator.strip()[:200], reason, deleted_at),
        )
    return {"feedback_id": feedback_id, "status": "deleted", "audit_id": audit_id, "deleted_at": deleted_at}
