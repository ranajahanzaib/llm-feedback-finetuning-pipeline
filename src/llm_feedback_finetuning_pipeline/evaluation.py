from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
import random
import resource
import statistics
import time
from typing import Any

from .datasets import verify_dataset_artifacts
from .db import new_id, now, path_for, transaction
from .tasks.products import ProductAttributes, ProductExtractionTask

FIELDS = tuple(ProductAttributes.model_fields)


def score_records(ground_truth: list[dict[str, Any]], predictions: list[Any]) -> dict[str, Any]:
    if len(ground_truth) != len(predictions):
        raise ValueError("ground_truth and predictions must have equal lengths")
    n = len(ground_truth)
    if n == 0:
        raise ValueError("evaluation set must not be empty")
    valid, exact, complete, json_valid = 0, 0, 0, 0
    field_correct = {field: 0 for field in FIELDS}
    examples = []
    baseline = []
    for truth, raw in zip(ground_truth, predictions, strict=True):
        truth = ProductExtractionTask.normalize(truth)
        raw_value = raw
        parse_success = False
        try:
            if isinstance(raw, str):
                raw = json.loads(raw)
            parse_success = True
            json_valid += 1
            if not isinstance(raw, dict) or set(raw) != set(FIELDS):
                raise ValueError("prediction must include every schema field and no extras")
            pred = ProductExtractionTask.normalize(raw)
            valid += 1
            complete += 1
        except Exception:
            pred = None
        if pred is not None:
            correct = all(pred[field] == truth.get(field) for field in FIELDS)
            exact += int(correct)
            for field in FIELDS:
                field_correct[field] += int(pred[field] == truth.get(field))
        else:
            correct = False
        if pred is None:
            error_categories = ["invalid_schema" if parse_success else "invalid_json"]
        else:
            error_categories = []
            for field in FIELDS:
                if pred[field] != truth.get(field):
                    error_categories.append(
                        "invented_attribute"
                        if truth.get(field) is None and pred[field] is not None
                        else "missed_attribute"
                        if truth.get(field) is not None and pred[field] is None
                        else f"wrong_{field}"
                    )
        baseline.append(correct)
        examples.append(
            {
                "ground_truth": truth,
                "raw_prediction": raw_value,
                "prediction": pred,
                "correct": correct,
                "error_category": None if correct else error_categories[0],
                "error_categories": error_categories,
            }
        )
    return {
        "n": n,
        "valid_json_rate": json_valid / n,
        "schema_adherence": valid / n,
        "required_field_completeness": complete / n,
        "exact_match_accuracy": exact / n,
        "per_field_accuracy": {k: v / n for k, v in field_correct.items()},
        "macro_field_accuracy": statistics.mean(field_correct.values()) / n,
        "examples": examples,
        "correct_flags": baseline,
    }


def compare(
    ground_truth: list[dict[str, Any]],
    baseline_predictions: list[Any],
    candidate_predictions: list[Any],
    *,
    baseline_id: str,
    candidate_id: str,
    dataset_id: str | None = None,
) -> dict[str, Any]:
    base = score_records(ground_truth, baseline_predictions)
    cand = score_records(ground_truth, candidate_predictions)
    improved = regressed = both_correct = both_wrong = 0
    cases = []
    for index, (b, c) in enumerate(zip(base["correct_flags"], cand["correct_flags"], strict=True)):
        improved += int(not b and c)
        regressed += int(b and not c)
        both_correct += int(b and c)
        both_wrong += int(not b and not c)
        cases.append(
            {
                "index": index,
                "baseline_correct": b,
                "candidate_correct": c,
                "outcome": "improved" if c and not b else "regressed" if b and not c else "unchanged",
            }
        )
    report = {
        "evaluation_id": new_id("ev"),
        "baseline_id": baseline_id,
        "candidate_id": candidate_id,
        "dataset_id": dataset_id,
        "created_at": now(),
        "baseline": base,
        "candidate": cand,
        "accuracy_delta": cand["exact_match_accuracy"] - base["exact_match_accuracy"],
        "improved": improved,
        "regressed": regressed,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "regression_rate_of_baseline_correct": regressed / (regressed + both_correct)
        if regressed + both_correct
        else None,
        "cases": cases,
        "sample_warning": "Evaluation set is small; conclusions are descriptive." if len(ground_truth) < 100 else None,
    }
    differences = [int(c) - int(b) for b, c in zip(base["correct_flags"], cand["correct_flags"], strict=True)]
    bootstrap_rng = random.Random(42)
    bootstrap_deltas = sorted(
        sum(bootstrap_rng.choice(differences) for _ in differences) / len(differences) for _ in range(5000)
    )
    report["paired_statistics"] = {
        "method": "paired bootstrap percentile interval",
        "resamples": 5000,
        "seed": 42,
        "accuracy_difference": report["accuracy_delta"],
        "confidence_level": 0.95,
        "confidence_interval": [bootstrap_deltas[124], bootstrap_deltas[4874]],
        "fragile": len(ground_truth) < 100,
    }
    artifact_dir = path_for("evaluations", report["evaluation_id"])
    artifact_dir.mkdir(parents=True, exist_ok=False)
    (artifact_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with transaction() as con:
        con.execute(
            "INSERT INTO evaluations VALUES(?,?,?,?,?)",
            (
                report["evaluation_id"],
                baseline_id,
                candidate_id,
                json.dumps(report),
                report["created_at"],
            ),
        )
        con.execute(
            "UPDATE models SET evaluation_id=?,metrics=?,status='evaluated' WHERE id=?",
            (report["evaluation_id"], json.dumps(cand), candidate_id),
        )
    return report


def run_mlx_evaluation(dataset_id: str, candidate_id: str, evaluation_set_id: str | None = None) -> dict[str, Any]:
    """Run paired greedy inference against the immutable dataset test split."""
    from pathlib import Path

    from mlx_lm import generate, load
    from mlx_lm.generate import make_sampler

    verify_dataset_artifacts(dataset_id)
    from .evaluation_sets import get_evaluation_set, verify_evaluation_set
    from .inference import _INFERENCE_LOCK
    from .registry import list_models

    with transaction() as con:
        dataset = con.execute("SELECT artifact_path,manifest FROM datasets WHERE id=?", (dataset_id,)).fetchone()
        prior_reports = [json.loads(row[0]) for row in con.execute("SELECT report FROM evaluations")]
    if dataset is None:
        raise KeyError(dataset_id)
    candidate = next((item for item in list_models() if item["id"] == candidate_id), None)
    if candidate is None:
        raise KeyError(candidate_id)
    if candidate.get("dataset_id") and candidate["dataset_id"] != dataset_id:
        raise ValueError("Evaluation dataset must match the candidate's training dataset lineage")
    if not candidate.get("adapter_path"):
        raise ValueError("Candidate has no trained adapter artifact")
    adapter_dir = path_for("models", candidate["adapter_path"])
    if not adapter_dir.is_dir() or not any(adapter_dir.glob("*.safetensors")):
        raise ValueError("Candidate adapter artifact is missing")
    expected_adapter_hash = (candidate.get("metrics") or {}).get("adapter_sha256")
    from .registry import adapter_checksum

    if not expected_adapter_hash or adapter_checksum(adapter_dir) != expected_adapter_hash:
        raise ValueError("Candidate adapter checksum does not match its registered artifact")
    test_path = Path(dataset["artifact_path"]) / "test.jsonl"
    evaluation_set: dict[str, Any] | None = None
    if evaluation_set_id:
        verify_evaluation_set(evaluation_set_id)
        evaluation_set = get_evaluation_set(evaluation_set_id)
        test_path = Path(evaluation_set["artifact_path"]) / "test.jsonl"
    examples = [json.loads(line) for line in test_path.read_text().splitlines() if line]
    if not examples:
        raise ValueError("Frozen test split is empty; build data with an independent test group")
    truth = [json.loads(row["messages"][2]["content"]) for row in examples]
    chats = [row["messages"][:2] for row in examples]
    train_path = Path(dataset["artifact_path"]) / "train.jsonl"
    train_examples = [json.loads(line) for line in train_path.read_text().splitlines() if line][:2]

    def run_model(adapter: str | None, *, few_shot: bool = False) -> tuple[list[str], list[float], float, int]:
        kwargs: dict[str, Any] = {}
        if adapter:
            kwargs["adapter_path"] = str(path_for("models", adapter))
        load_started = time.perf_counter()
        model, tokenizer = load(candidate["base_model"], **kwargs)
        load_ms = (time.perf_counter() - load_started) * 1000
        raw_results, timings = [], []
        for messages in chats:
            if few_shot and train_examples:
                examples_text = "\n".join(
                    "Input: "
                    + example["messages"][1]["content"][:500]
                    + "\nOutput: "
                    + example["messages"][2]["content"][:500]
                    for example in train_examples
                )
                messages = [
                    {
                        "role": "system",
                        "content": ProductExtractionTask.system_prompt
                        + "\nDemonstrations from training data:\n"
                        + examples_text,
                    },
                    messages[1],
                ]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            started = time.perf_counter()
            raw_results.append(
                generate(
                    model,
                    tokenizer,
                    prompt=prompt,
                    max_tokens=256,
                    sampler=make_sampler(temp=0.0),
                )
            )
            timings.append((time.perf_counter() - started) * 1000)
        return raw_results, timings, load_ms, sum(len(tokenizer.encode(result)) for result in raw_results)

    with _INFERENCE_LOCK:
        baseline_raw, baseline_times, baseline_load_ms, baseline_tokens = run_model(None)
        fewshot_raw, fewshot_times, fewshot_load_ms, fewshot_tokens = run_model(None, few_shot=True)
        candidate_raw, candidate_times, candidate_load_ms, candidate_tokens = run_model(candidate["adapter_path"])
    report = compare(
        truth,
        baseline_raw,
        candidate_raw,
        baseline_id="untuned:" + candidate["base_model"],
        candidate_id=candidate_id,
        dataset_id=dataset_id,
    )
    for index, example in enumerate(examples):
        input_text = example["messages"][1]["content"]
        report["baseline"]["examples"][index]["input_text"] = input_text
        report["candidate"]["examples"][index]["input_text"] = input_text
    report["baseline"]["latency_ms_median"] = statistics.median(baseline_times)
    report["candidate"]["latency_ms_median"] = statistics.median(candidate_times)
    report["baseline"]["latency_ms_p95"] = sorted(baseline_times)[
        min(len(baseline_times) - 1, math.ceil(len(baseline_times) * 0.95) - 1)
    ]
    report["candidate"]["latency_ms_p95"] = sorted(candidate_times)[
        min(len(candidate_times) - 1, math.ceil(len(candidate_times) * 0.95) - 1)
    ]
    report["baseline"]["cold_load_ms"] = baseline_load_ms
    report["candidate"]["cold_load_ms"] = candidate_load_ms
    report["baseline"]["generated_tokens"] = baseline_tokens
    report["candidate"]["generated_tokens"] = candidate_tokens
    report["baseline"]["tokens_per_second"] = (
        baseline_tokens / (sum(baseline_times) / 1000) if sum(baseline_times) else None
    )
    report["candidate"]["tokens_per_second"] = (
        candidate_tokens / (sum(candidate_times) / 1000) if sum(candidate_times) else None
    )
    report["observed_process_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report["benchmark_assumptions"] = {
        "request_concurrency": 1,
        "latency_scope": "sequential generation; cold model load measured separately",
        "peak_memory_scope": "process maximum RSS; MLX unified-memory allocations may not be fully represented",
        "power_draw_watts": None,
        "cost_per_correct_task": None,
        "cost_note": "Unavailable without measured power and configured hardware amortization assumptions.",
    }
    report["baseline"]["raw_predictions"] = baseline_raw
    report["candidate"]["raw_predictions"] = candidate_raw
    report["candidate"]["adapter_artifact_bytes"] = sum(
        path.stat().st_size for path in adapter_dir.glob("*.safetensors")
    )
    fewshot_metrics = score_records(truth, fewshot_raw)
    base_flags = report["baseline"]["correct_flags"]
    fewshot_flags = fewshot_metrics["correct_flags"]
    report["prompting_only"] = {
        **{key: value for key, value in fewshot_metrics.items() if key != "correct_flags"},
        "model_id": "untuned:" + candidate["base_model"],
        "strategy": "two demonstrations sampled from this dataset's training split",
        "baseline_accuracy_delta": fewshot_metrics["exact_match_accuracy"] - report["baseline"]["exact_match_accuracy"],
        "improved": sum(not base and shot for base, shot in zip(base_flags, fewshot_flags, strict=True)),
        "regressed": sum(base and not shot for base, shot in zip(base_flags, fewshot_flags, strict=True)),
        "latency_ms_median": statistics.median(fewshot_times),
        "cold_load_ms": fewshot_load_ms,
        "generated_tokens": fewshot_tokens,
        "tokens_per_second": fewshot_tokens / (sum(fewshot_times) / 1000) if sum(fewshot_times) else None,
        "raw_predictions": fewshot_raw,
    }
    report["evaluation_mode"] = "real_mlx_greedy_inference"
    report["prompt_version"] = json.loads(dataset["manifest"]).get("prompt_version", "unknown")
    report["evaluation_config"] = {
        "decoding": "greedy",
        "temperature": 0.0,
        "max_tokens": 256,
        "stop_criteria": None,
        "prompting_only_demonstrations": min(2, len(train_examples)),
        "prompting_only_source_split": "train",
        "pairing": "same immutable test records and scoring implementation",
        "dataset_manifest_sha256": hashlib.sha256(dataset["manifest"].encode()).hexdigest(),
        "evaluation_set_manifest_sha256": hashlib.sha256(
            json.dumps(evaluation_set["manifest"], sort_keys=True).encode()
        ).hexdigest()
        if evaluation_set
        else None,
        "mlx_lm_version": importlib.metadata.version("mlx-lm"),
        "python_version": platform.python_version(),
        "resolved_base_model_revision": candidate.get("metrics", {}).get("resolved_base_model_revision"),
    }
    prior_test_runs = sum(
        item.get("evaluation_set_id") == evaluation_set_id
        if evaluation_set_id
        else item.get("dataset_id") == dataset_id and not item.get("evaluation_set_id")
        for item in prior_reports
    )
    report["evaluation_set_id"] = evaluation_set_id
    report["evaluation_source"] = "independent_frozen_set" if evaluation_set else "dataset_test_split"
    report["evaluation_reuse_count"] = prior_test_runs
    report["test_set_warning"] = (
        "This frozen evaluation data was used previously; repeated candidate comparison can overfit to it."
        if prior_test_runs
        else None
    )
    with transaction() as con:
        con.execute(
            "UPDATE evaluations SET report=? WHERE id=?",
            (json.dumps(report), report["evaluation_id"]),
        )
        con.execute(
            "UPDATE models SET metrics=? WHERE id=?",
            (json.dumps(report["candidate"]), candidate_id),
        )
    (path_for("evaluations", report["evaluation_id"]) / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report
