from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from llm_feedback_finetuning_pipeline import config
from llm_feedback_finetuning_pipeline.benchmarking import local_cost_scenarios
from llm_feedback_finetuning_pipeline.datasets import (
    build_dataset,
    diff_datasets,
    list_datasets,
    verify_dataset_artifacts,
)
from llm_feedback_finetuning_pipeline.db import init_db
from llm_feedback_finetuning_pipeline.evaluation import compare, score_records
from llm_feedback_finetuning_pipeline.evaluation_sets import (
    create_evaluation_set,
    list_evaluation_sets,
    verify_evaluation_set,
)
from llm_feedback_finetuning_pipeline.feedback import (
    delete_feedback,
    import_records,
    list_feedback,
    redact_feedback,
    resolve_conflict,
    review,
)
from llm_feedback_finetuning_pipeline.registry import (
    active_model,
    deployments,
    gate,
    list_models,
    promote,
    register_model,
    rollback,
)
from llm_feedback_finetuning_pipeline.tasks import ProductExtractionTask, TaskSpec
from llm_feedback_finetuning_pipeline.tasks.products import ProductAttributes


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "settings", config.Settings(data_dir=tmp_path / "home"))
    # Modules import the settings singleton, replace the bound objects as well.
    import llm_feedback_finetuning_pipeline.db as db
    import llm_feedback_finetuning_pipeline.registry as registry

    monkeypatch.setattr(db, "settings", config.settings)
    monkeypatch.setattr(registry, "path_for", db.path_for)
    init_db()


def item(text: str, category: str = "shirt", group: str | None = None):
    return {
        "input_text": text,
        "corrected_output": {
            "category": category,
            "gender": None,
            "color": "white",
            "size": "m",
            "material": "cotton",
            "waterproof": None,
        },
        "source_group_id": group,
    }


def test_product_schema_unknown_fields_and_normalization():
    assert isinstance(ProductExtractionTask, TaskSpec)
    with pytest.raises(ValidationError):
        ProductAttributes.model_validate({"fabric": "cotton"})
    normalized = ProductExtractionTask.normalize({"category": " Shirt ", "color": " NAVY   BLUE "})
    assert normalized["category"] == "shirt"
    assert normalized["color"] == "navy blue"
    assert normalized["waterproof"] is None
    assert "waterproof" in ProductExtractionTask.system_prompt
    assert "Do not add markdown fences" in ProductExtractionTask.system_prompt


def test_import_defaults_pending_detects_duplicates_conflicts_and_privacy():
    records = [
        item("white cotton shirt"),
        item("white cotton shirt"),
        item("white cotton shirt", "coat"),
        item("Email a@b.com"),
    ]
    result = import_records(records, commit=True)
    assert result["valid_count"] == 4
    assert len(result["duplicates"]) == 1
    assert len(result["conflicts"]) == 1
    assert result["privacy_warnings"]
    states = {entry["status"] for entry in list_feedback()}
    assert "pending" in states and "quarantined" in states


def test_import_reports_near_duplicates_and_placeholder_labels():
    first = item("White cotton hiking shirt size medium for trail use")
    second = item("White cotton hiking shirts size medium for trail use")
    second["corrected_output"]["category"] = "placeholder"
    result = import_records([first, second])
    assert result["near_duplicates"]
    assert result["quality_warnings"][0]["warnings"] == ["placeholder_label:category"]
    normalized = ProductExtractionTask.normalize({"category": "unknown"})
    assert normalized["category"] is None


def test_review_transition_is_audited_and_terminal():
    import_records([item("cotton shirt")], commit=True)
    feedback_id = list_feedback()[0]["id"]
    reviewed = review(feedback_id, "approved", "operator", "checked")
    assert reviewed["status"] == "approved"
    with pytest.raises(ValueError, match="cannot review"):
        review(feedback_id, "rejected", "operator")


def test_database_conflict_quarantines_both_and_requires_adjudication():
    import_records([item("same exact catalog text")], commit=True)
    original = list_feedback()[0]
    review(original["id"], "approved", "operator")
    conflicting = import_records([item("same exact catalog text", category="coat")], commit=True)
    assert conflicting["database_conflicts"]
    records = {row["id"]: row for row in list_feedback()}
    assert records[original["id"]]["status"] == "quarantined"
    selected_id = conflicting["feedback_ids"][0]
    assert records[selected_id]["status"] == "quarantined"
    with pytest.raises(ValueError, match="reason"):
        resolve_conflict(selected_id, "operator", " ")
    selected = resolve_conflict(selected_id, "operator", "Catalog manager verified source listing")
    assert selected["status"] == "approved"
    assert selected["conflict"] == 0


def test_sensitive_input_can_be_redacted_before_approval():
    record_data = item("Contact buyer at person@example.com")
    record_data["notes"] = "Follow up with person@example.com"
    record_data["model_generation_metadata"] = {"credential": "api" + "_key=TEST_ONLY_SENTINEL"}
    import_records([record_data], commit=True)
    record = list_feedback()[0]
    with pytest.raises(ValueError, match="privacy-flagged"):
        review(record["id"], "approved", "operator")
    cleaned = redact_feedback(record["id"], "operator")
    assert "person@example.com" not in cleaned["input_text"]
    assert "person@example.com" not in cleaned["notes"]
    assert "TEST_ONLY_SENTINEL" not in json.dumps(cleaned["model_generation_metadata"])
    assert cleaned["privacy_flags"] == []
    approved = review(record["id"], "approved", "operator")
    assert approved["status"] == "approved"


def test_dataset_immutable_reproducible_and_group_aware():
    import_records([item(f"product text {i}", group=f"group-{i // 2}") for i in range(12)], commit=True)
    for record in list_feedback():
        review(record["id"], "approved", "operator")
    first = build_dataset(seed=19)
    second = build_dataset(seed=19)
    assert first["counts"] == second["counts"]
    assert first["file_sha256"] == second["file_sha256"]
    assert first["statistics"]["approved_examples"] == 12
    assert first["split_feedback_ids"]["test"]
    assert first["test_input_hashes"]
    assert first["contamination"]["previous_frozen_test_overlap_count"] == 0
    assert diff_datasets(first["dataset_id"], second["dataset_id"])["record_count_delta"] == 0
    assert (
        diff_datasets(first["dataset_id"], second["dataset_id"])["contamination"]["previous_frozen_test_overlap_count"]
        > 0
    )
    versions = list_datasets()
    assert len(versions) == 2
    for item_ in versions:
        manifest = item_["manifest"]
        assert sum(manifest["counts"].values()) == 12
        assert set(("train", "valid", "test")) == set(manifest["counts"])
        assert all((Path(item_["artifact_path"]) / f"{split}.jsonl").exists() for split in manifest["counts"])
    source_group = {row["input_text"]: row["source_group_id"] for row in list_feedback()}
    assigned = {}
    for item_ in versions[:1]:
        for split in ("train", "valid", "test"):
            rows = [
                json.loads(line) for line in (Path(item_["artifact_path"]) / f"{split}.jsonl").read_text().splitlines()
            ]
            for row in rows:
                text = row["messages"][1]["content"]
                group = source_group[text]
                assert group not in assigned or assigned[group] == split
                assigned[group] = split


def test_dataset_artifact_integrity_is_checked(tmp_path):
    import_records([item(f"integrity product {i}") for i in range(4)], commit=True)
    for record in list_feedback():
        review(record["id"], "approved", "operator")
    manifest = build_dataset()
    assert len(verify_dataset_artifacts(manifest["dataset_id"])["verified_files"]) == 3
    dataset = next(item for item in list_datasets() if item["id"] == manifest["dataset_id"])
    train_path = Path(dataset["artifact_path"]) / "train.jsonl"
    train_path.write_text(train_path.read_text() + "{}\n")
    with pytest.raises(ValueError, match="integrity check failed"):
        verify_dataset_artifacts(manifest["dataset_id"])


def test_feedback_deletion_erases_source_and_blocks_snapshot_references():
    import_records([item(f"deletable product {i}") for i in range(4)], commit=True)
    rows = list_feedback()
    deleted = delete_feedback(rows[0]["id"], "operator", "User deletion request")
    assert deleted["status"] == "deleted"
    assert rows[0]["id"] not in {row["id"] for row in list_feedback()}
    for row in list_feedback():
        review(row["id"], "approved", "operator")
    dataset = build_dataset()
    with pytest.raises(ValueError, match="immutable dataset"):
        delete_feedback(dataset["selected_feedback_ids"][0], "operator", "Erase source")


def test_independent_evaluation_set_is_frozen_and_kept_out_of_future_training():
    records = [
        item("Men's waterproof trail jacket, red", "jacket"),
        item("Children's rubber rain boots, yellow", "boot"),
        item("Wool baseball cap, navy", "hat"),
    ]
    manifest = create_evaluation_set(records)
    assert manifest["count"] == 3
    assert not list_feedback()
    assert verify_evaluation_set(manifest["evaluation_set_id"])["verified_files"] == ["test.jsonl"]
    assert len(list_evaluation_sets()) == 1

    training_records = [item(f"Training catalog example number {i}") for i in range(4)]
    import_records(training_records, commit=True)
    for record in list_feedback():
        review(record["id"], "approved", "operator")
    leaked_record = {**records[0], "input_text": "  " + records[0]["input_text"].replace("waterproof", "waterproof  ")}
    leaked = import_records([leaked_record], commit=True)
    leaked_id = leaked["feedback_ids"][0]
    review(leaked_id, "approved", "operator")
    dataset = build_dataset()
    assert leaked_id in dataset["excluded_frozen_evaluation_feedback_ids"]
    assert leaked_id not in dataset["selected_feedback_ids"]


def test_independent_evaluation_set_rejects_existing_feedback_overlap():
    record = item("Same catalog example in source systems")
    import_records([record], commit=True)
    records = [record, item("Distinct snow pants and jacket"), item("Ceramic travel mug accessory")]
    with pytest.raises(ValueError, match="overlap existing feedback"):
        create_evaluation_set(records)


def test_evaluation_metrics_regressions_and_empty_set():
    truth = [
        {
            "category": "shirt",
            "gender": None,
            "color": None,
            "size": "M",
            "material": None,
            "waterproof": None,
        }
    ]
    perfect = dict(truth[0])
    wrong = {**perfect, "category": "coat"}
    assert score_records(truth, [perfect])["exact_match_accuracy"] == 1
    assert score_records(truth, [{**perfect, "size": "m"}])["exact_match_accuracy"] == 1
    report = compare(truth, [perfect], [wrong], baseline_id="base", candidate_id="candidate")
    assert report["regressed"] == 1
    assert report["regression_rate_of_baseline_correct"] == 1
    assert report["paired_statistics"]["fragile"] is True
    assert len(report["paired_statistics"]["confidence_interval"]) == 2
    assert report["candidate"]["examples"][0]["error_categories"] == ["wrong_category"]
    incomplete = {key: value for key, value in perfect.items() if key != "gender"}
    incomplete_score = score_records(truth, [incomplete])
    assert incomplete_score["schema_adherence"] == 0
    assert incomplete_score["required_field_completeness"] == 0
    with pytest.raises(ValueError, match="must not be empty"):
        score_records([], [])


def test_local_cost_requires_assumptions_and_amortizes_training():
    missing = local_cost_scenarios(
        accuracy=0.8,
        power_watts=None,
        electricity_usd_per_kwh=0.2,
        inference_seconds=1,
        training_cost_usd=12,
        monthly_hardware_cost_usd=20,
    )
    assert not missing["available"]
    costs = local_cost_scenarios(
        accuracy=0.8,
        power_watts=100,
        electricity_usd_per_kwh=0.2,
        inference_seconds=1,
        training_cost_usd=12,
        monthly_hardware_cost_usd=20,
        training_amortization_months=12,
    )
    assert costs["available"]
    assert costs["scenarios"]["1000"]["monthly_requests"] == 1000
    assert costs["scenarios"]["100000"]["monthly_cost_usd"] > costs["scenarios"]["1000"]["monthly_cost_usd"]


def test_promotion_requires_evaluation_and_artifacts():
    model = register_model("local-base", metrics={"n": 2, "schema_adherence": 1})
    decision = gate(model)
    assert not decision["eligible"]
    assert not decision["checks"]["evaluation_attached"]["pass"]


def test_promotion_and_rollback_restore_registry_selection(monkeypatch):
    import llm_feedback_finetuning_pipeline.inference as inference

    monkeypatch.setattr(inference, "predict", lambda *args, **kwargs: {"validation": "valid"})
    truth = [
        {
            "category": "shirt",
            "gender": None,
            "color": "white",
            "size": "M",
            "material": "cotton",
            "waterproof": None,
        }
        for _ in range(30)
    ]
    first = register_model("base-model")
    compare(truth, truth, truth, baseline_id="base", candidate_id=first["id"])
    promote(first["id"], "operator", "unit test deployment override", override=True)
    assert active_model()["id"] == first["id"]

    second = register_model("base-model")
    compare(truth, truth, truth, baseline_id="base", candidate_id=second["id"])
    promote(second["id"], "operator", "unit test deployment override", override=True)
    assert active_model()["id"] == second["id"]

    result = rollback("operator", "restore previous deployment")
    assert result["active_model"] == first["id"]
    assert active_model()["id"] == first["id"]
    assert len(deployments()) == 3
    assert {model["id"]: model["status"] for model in list_models()} == {
        first["id"]: "production",
        second["id"]: "archived",
    }


def test_api_inference_and_admin_auth(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import llm_feedback_finetuning_pipeline.api as api
    import llm_feedback_finetuning_pipeline.db as db

    test_api_key = "TEST_ONLY" + "_KEY_NOT_SECRET"
    isolated = config.Settings(data_dir=tmp_path / "api-home", api_key=test_api_key)
    monkeypatch.setattr(api, "settings", isolated)
    monkeypatch.setattr(db, "settings", isolated)
    with TestClient(api.app) as client:
        response = client.post(
            "/api/v1/tasks/product-attributes/predict",
            json={"input_text": "White cotton shirt, size M"},
        )
        assert response.status_code == 200
        assert response.json()["mode"] == "simulated"
        no_fallback = client.post(
            "/api/v1/tasks/product-attributes/predict",
            json={"input_text": "shirt", "mode": "mlx"},
        )
        assert no_fallback.status_code == 503
        assert no_fallback.json()["detail"]["error"] == "no_active_model"
        assert client.get("/api/v1/feedback").status_code == 401
        assert client.get("/api/v1/feedback", headers={"X-LLM-Feedback-Key": test_api_key}).status_code == 200
        feedback_response = client.post(
            "/api/v1/feedback",
            headers={"X-LLM-Feedback-Key": test_api_key},
            json={
                "input_text": "White cotton shirt, size M",
                "corrected_output": {
                    "category": "shirt",
                    "gender": None,
                    "color": "white",
                    "size": "M",
                    "material": "cotton",
                    "waterproof": None,
                },
                "original_model_id": "model-before",
                "prompt_version": "product-json-v2",
                "model_generation_metadata": {"temperature": 0, "tokens": 42},
            },
        )
        assert feedback_response.status_code == 200, feedback_response.text
        saved = feedback_response.json()["feedback"]
        assert saved["original_model_id"] == "model-before"
        assert saved["prompt_version"] == "product-json-v2"
        assert saved["model_generation_metadata"] == {"temperature": 0, "tokens": 42}
        feedback_id = saved["id"]
        assert (
            client.get(f"/api/v1/feedback/{feedback_id}", headers={"X-LLM-Feedback-Key": test_api_key}).status_code
            == 200
        )
        assert (
            client.get("/api/v1/feedback?source=api", headers={"X-LLM-Feedback-Key": test_api_key}).status_code
            == 200
        )
        bulk_payload = {"feedback_ids": [feedback_id], "decision": "approved", "reviewer": "api-test"}
        bulk_url = "/api/v1/feedback/review/bulk"
        headers = {"X-LLM-Feedback-Key": test_api_key}
        assert client.post(bulk_url, headers=headers, json=bulk_payload).status_code == 409
        bulk_payload["confirm"] = True
        assert client.post(bulk_url, headers=headers, json=bulk_payload).status_code == 200
        cost = client.post("/api/v1/benchmarks/cost", headers=headers, json={"accuracy": 0.9})
        assert cost.status_code == 200
        assert cost.json()["available"] is False
        evaluation_records = [
            item("Men's waterproof trail jacket, red", "jacket"),
            item("Children's rubber rain boots, yellow", "boot"),
            item("Wool baseball cap, navy", "hat"),
        ]
        upload_body = "\n".join(json.dumps(record) for record in evaluation_records).encode()
        frozen = client.post(
            "/api/v1/evaluation-sets",
            headers=headers,
            files={"file": ("holdout.jsonl", upload_body, "application/jsonl")},
        )
        assert frozen.status_code == 200, frozen.text
        assert frozen.json()["count"] == 3
        assert len(client.get("/api/v1/evaluation-sets", headers=headers).json()) == 1
