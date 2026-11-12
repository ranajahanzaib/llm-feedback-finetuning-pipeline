from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from .config import settings


def now() -> str:
    return datetime.now(UTC).isoformat()


def connect() -> sqlite3.Connection:
    settings.initialize()
    con = sqlite3.connect(settings.database_path, timeout=30)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys=ON")
    return con


@contextmanager
def transaction() -> Iterator[sqlite3.Connection]:
    con = connect()
    try:
        with con:
            yield con
    finally:
        con.close()


def init_db() -> None:
    with transaction() as con:
        con.executescript("""
        CREATE TABLE IF NOT EXISTS feedback (
          id TEXT PRIMARY KEY, task_id TEXT NOT NULL, schema_version TEXT NOT NULL,
          input_text TEXT NOT NULL, original_prediction TEXT, corrected_output TEXT NOT NULL,
          source TEXT NOT NULL, source_record_id TEXT, source_group_id TEXT,
          original_model_id TEXT, prompt_version TEXT, model_generation_metadata TEXT,
          reviewer TEXT, status TEXT NOT NULL CHECK(status IN ('pending','approved','rejected','quarantined')),
          notes TEXT, privacy_flags TEXT NOT NULL DEFAULT '[]', created_at TEXT NOT NULL,
          reviewed_at TEXT, duplicate_of TEXT, conflict INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS review_history (
          id TEXT PRIMARY KEY, feedback_id TEXT NOT NULL REFERENCES feedback(id),
          old_status TEXT NOT NULL, new_status TEXT NOT NULL, reviewer TEXT, notes TEXT,
          changed_at TEXT NOT NULL, prior_output TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS deletion_audit (
          id TEXT PRIMARY KEY, feedback_id TEXT NOT NULL, previous_status TEXT NOT NULL,
          operator TEXT NOT NULL, reason TEXT NOT NULL, deleted_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS datasets (
          id TEXT PRIMARY KEY, task_id TEXT NOT NULL, manifest TEXT NOT NULL,
          artifact_path TEXT NOT NULL, created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS evaluation_sets (
          id TEXT PRIMARY KEY, task_id TEXT NOT NULL, manifest TEXT NOT NULL,
          artifact_path TEXT NOT NULL, created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS experiments (
          id TEXT PRIMARY KEY, dataset_id TEXT, status TEXT NOT NULL, config TEXT NOT NULL,
          manifest TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS evaluations (
          id TEXT PRIMARY KEY, baseline_id TEXT NOT NULL, candidate_id TEXT NOT NULL,
          report TEXT NOT NULL, created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS models (
          id TEXT PRIMARY KEY, base_model TEXT NOT NULL, adapter_path TEXT,
          dataset_id TEXT, experiment_id TEXT, evaluation_id TEXT, status TEXT NOT NULL,
          metrics TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS deployments (
          id TEXT PRIMARY KEY, model_id TEXT NOT NULL, action TEXT NOT NULL,
          operator TEXT, reason TEXT, created_at TEXT NOT NULL, details TEXT NOT NULL DEFAULT '{}'
        );
        CREATE TABLE IF NOT EXISTS app_state (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        """)
        columns = {row[1] for row in con.execute("PRAGMA table_info(feedback)")}
        for column in ("original_model_id", "prompt_version", "model_generation_metadata"):
            if column not in columns:
                con.execute(f"ALTER TABLE feedback ADD COLUMN {column} TEXT")
        deployment_columns = {row[1] for row in con.execute("PRAGMA table_info(deployments)")}
        if "details" not in deployment_columns:
            con.execute("ALTER TABLE deployments ADD COLUMN details TEXT NOT NULL DEFAULT '{}' ")


def asdict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    result = dict(row)
    for key in (
        "corrected_output",
        "original_prediction",
        "model_generation_metadata",
        "privacy_flags",
        "manifest",
        "config",
        "metrics",
        "report",
    ):
        if key in result and result[key] is not None:
            try:
                result[key] = json.loads(result[key])
            except (ValueError, TypeError):
                pass
    return result


def path_for(*parts: str) -> Path:
    root = settings.artifacts.resolve()
    target = root.joinpath(*parts).resolve()
    if not target.is_relative_to(root):
        raise ValueError("artifact path escapes configured artifact directory")
    return target


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex[:12]}"
