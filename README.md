# LLM Feedback Fine-Tuning Pipeline

**Turn business corrections into specialized local AI models.** LLM Feedback Fine-Tuning Pipeline is a local-first workflow for reviewing AI corrections, creating traceable datasets, adapting compact MLX models, measuring regressions, and gating deployment.

[![CI](https://github.com/ranajahanzaib/llm-feedback-finetuning-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/ranajahanzaib/llm-feedback-finetuning-pipeline/actions/workflows/ci.yml) [![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

LLM Feedback Fine-Tuning Pipeline provides a persistent feedback workflow, immutable dataset snapshots, a localhost API, a React operator console, and an explicitly labeled deterministic demo mode. Real MLX training and inference require compatible Apple Silicon hardware and model weights. Synthetic fixtures exercise the workflow and do not establish model quality.

**Project status: Alpha.** Interfaces and artifact formats may change before a 1.0 release.

## Workflow

```text
Product text → prediction → human correction → review → versioned dataset
       → local LoRA experiment → frozen evaluation → promotion decision → API
```

The implemented task extracts category, gender, color, size, material, and waterproof status. Missing information maps to `null`; attributes must not be guessed. Imported feedback defaults to **pending**. The demo fixture is synthetic and marked pre-approved.

`llm_feedback_finetuning_pipeline.tasks.TaskSpec` defines the task contract: identifiers and prompt version, a Pydantic output model, input validation, label normalization, and MLX-LM example serialization. Product attribute extraction is the supported task. Additional tasks require task-specific metrics and API routes.

## Requirements and setup

- Python 3.11–3.13 and [uv](https://docs.astral.sh/uv/)
- Node.js 20+ and pnpm for the dashboard
- macOS on Apple Silicon plus MLX and model weights for actual model inference/training

```bash
git clone https://github.com/ranajahanzaib/llm-feedback-finetuning-pipeline.git
cd llm-feedback-finetuning-pipeline
uv sync --extra dev
uv run llm-feedback-finetuning-pipeline init
uv run llm-feedback-finetuning-pipeline doctor
```

Optional MLX dependencies:

```bash
uv sync --extra mlx --extra dev
```

The default model reference is `Qwen/Qwen3-0.6B-MLX-bf16`. The app does not automatically fetch model weights. MLX-LM model support and CLI options can change between releases; run `llm-feedback-finetuning-pipeline doctor` and verify the selected model card before a real experiment. Set `LLM_FEEDBACK_FINETUNING_PIPELINE_MODEL` to a compatible local or published model reference.

The default model card declares Apache-2.0 for its weights ([model card](https://huggingface.co/Qwen/Qwen3-0.6B-MLX-bf16)). LLM Feedback Fine-Tuning Pipeline does not redistribute model weights. Other model references have independent licenses and revisions; verify their terms before commercial use. When available in the local Hugging Face cache, experiment and evaluation records capture the resolved snapshot revision.

## Lightweight demo

In one terminal:

```bash
uv run llm-feedback-finetuning-pipeline demo
uv run llm-feedback-finetuning-pipeline serve
```

In another:

```bash
cd web
pnpm install
pnpm run dev
```

Open the local Vite URL. The synthetic fixture contains 120 deterministic examples and is pre-approved solely to let an operator inspect dataset creation. Playground predictions in simulated mode are labeled and are not ML measurements. To create a dataset from fixture examples:

```bash
uv run llm-feedback-finetuning-pipeline dataset build
```

Review the manifest in `~/.llm-feedback-finetuning-pipeline/artifacts/datasets/<dataset-id>/manifest.json`. Each version includes MLX-LM chat `train.jsonl`, `valid.jsonl`, and `test.jsonl`, split counts, provenance, distributions, and SHA-256 hashes. Versions are stored as new directories and are never rewritten.

## Importing and reviewing corrections

CSV and JSONL columns use `input_text`, `corrected_output` (a JSON object or JSON-encoded CSV cell), and optional `original_prediction`, `source`, `source_record_id`, `source_group_id`, `original_model_id`, `prompt_version`, `model_generation_metadata`, and `notes`.

Preview an import without writing, then deliberately commit after reviewing the report:

```bash
uv run llm-feedback-finetuning-pipeline feedback import corrections.jsonl
uv run llm-feedback-finetuning-pipeline feedback import corrections.jsonl --commit
uv run llm-feedback-finetuning-pipeline feedback list --status pending
uv run llm-feedback-finetuning-pipeline feedback review fb-<id> approved --reviewer "catalog-ops"
```

Identical inputs with conflicting labels are quarantined. Review decisions are recorded in a history table. Basic email, phone, and credential-pattern checks are heuristics, not a guarantee of anonymization. Review and redact sensitive data before approving it.

To resolve an input conflict, inspect both quarantined records and explicitly adjudicate the selected one in the UI or through `POST /api/v1/feedback/{id}/resolve-conflict`. Privacy-flagged feedback cannot be approved until redacted. Eligible source records can be removed with `llm-feedback-finetuning-pipeline feedback delete <id> "reason"` or the API; deletion is blocked once an immutable dataset references the record. The deletion audit retains only the record ID, operator, reason, previous state, and timestamp.

The API also provides an upload preview at `POST /api/v1/import/preview`; add `?commit=true` to commit only if all records validate. The optional `LLM_FEEDBACK_FINETUNING_PIPELINE_API_KEY` protects administrative routes using the `X-LLM-Feedback-Key` header. Enter it in the dashboard header when enabled; the browser keeps it in session storage only. Set secrets in the process environment, not in tracked files. The server binds to `127.0.0.1` by default.

## Real local inference and training

Use actual MLX inference explicitly from the CLI. For the API/playground, set `LLM_FEEDBACK_FINETUNING_PIPELINE_MODE=mlx` before starting the server:

```bash
uv run llm-feedback-finetuning-pipeline predict --text "Men's waterproof hiking jacket, navy blue, size XL, polyester shell." --mode mlx
```

For training, install the MLX extra, ensure the model is available, create a dataset with enough approved examples, and start:

```bash
uv run llm-feedback-finetuning-pipeline train start <dataset-id> --preset smoke
uv run llm-feedback-finetuning-pipeline train status
uv run llm-feedback-finetuning-pipeline train status <experiment-id>
uv run llm-feedback-finetuning-pipeline train cancel <experiment-id>
```

Supported presets are `smoke`, `standard`, and `extended`. Smoke is an operational check, not a training-quality claim. The MLX-LM subprocess runs with argument arrays and stores logs and manifests below the LLM Feedback Fine-Tuning Pipeline artifact root. Quantization and QLoRA remain exploratory; the initial supported route is LoRA.

Training uses MLX-LM's `--mask-prompt` option for chat data so loss is computed on the assistant completion. Its config uses MLX-LM's `scale` field for LoRA; it does not use PEFT's `alpha` name. See the [MLX-LM LoRA guide](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md).

LoRA learns a small adapter while keeping the base weights fixed; QLoRA performs adapter training with a quantized base model when that model/runtime combination supports it. More corrections do not guarantee better results: label noise, duplicated examples, or a narrow dataset can hurt performance. The frozen test split is kept out of the training files and overlap is reported when a later dataset reuses a previous test input. Repeatedly comparing candidates on the same test split can still lead to tuning against that holdout, so a prompting-only baseline is useful when a few examples in the prompt match fine-tuning quality with less operational work.

## API

Start with `uv run llm-feedback-finetuning-pipeline serve`. Interactive OpenAPI docs are at `http://127.0.0.1:8000/docs`.

```bash
curl -s http://127.0.0.1:8000/api/v1/tasks/product-attributes/predict \
  -H 'Content-Type: application/json' \
  -d '{"input_text":"White cotton shirt, size M"}'
```

Other endpoints cover health/readiness, feedback and review, dataset creation, training job status/cancellation, registry, promotion checks, deployment history, and rollback. Run paired real inference on a dataset's frozen test split with:

```bash
uv run llm-feedback-finetuning-pipeline evaluate run <dataset-id> <candidate-model-id>
```

This records raw predictions, schema/exact-match/per-field metrics, paired improvements and regressions, a seeded paired bootstrap interval, model-load and warm-request latency, generated tokens, and process peak RSS. It also runs a two-example prompting-only baseline using examples from the training split. Greedy decoding is used; exact reproducibility across runtime versions is not guaranteed. Candidates without evidence are blocked by default.

## Evaluation and deployment integrity

`llm_feedback_finetuning_pipeline.evaluation` compares the untuned base model and trained adapter on the immutable test split, computes exact-match, per-field, schema adherence, and paired improved/regressed counts, and records raw predictions and median latency. It refuses empty sets and length mismatches, and makes the regression denominator explicit. Promotion requires attached evaluation evidence, at least 30 examples, and 98% schema adherence; override requires an explicit CLI/API option and is audited.

Do not treat a tiny synthetic holdout as production evidence. The hosted baseline remains unavailable unless an explicit provider integration is added. Peak RSS is process-wide and may not capture all MLX unified-memory allocations. Power draw must be measured or supplied by the operator; LLM Feedback Fine-Tuning Pipeline does not claim a cost when required assumptions are missing. Submit assumptions for 1k, 10k, and 100k monthly requests to `POST /api/v1/benchmarks/cost`; training cost is amortized over the configured number of months. The correct choice can be to reject a candidate or not fine-tune.

Dataset manifests record source coverage, missing-attribute counts, split feedback IDs, content hashes, and overlap with prior frozen test inputs. Training and evaluation verify the snapshot hashes before reading the data. Inspect a version with `llm-feedback-finetuning-pipeline dataset inspect <dataset-id>` and compare versions with `llm-feedback-finetuning-pipeline dataset diff <old-id> <new-id>`.

For an independent holdout, freeze a separate CSV or JSONL file with `llm-feedback-finetuning-pipeline evaluation-set freeze holdout.jsonl`. The API accepts the same file at `POST /api/v1/evaluation-sets`, and the dashboard has a holdout upload control. Inputs must not overlap feedback, prior training snapshots, or another frozen set. These examples are never written into feedback or training files; later matching approved feedback is excluded from future dataset builds. Evaluate with `llm-feedback-finetuning-pipeline evaluate run <training-dataset-id> <candidate-id> --evaluation-set <evaluation-set-id>`. The candidate must still belong to the selected training dataset.

## Privacy and limitations

- No telemetry or automatic upload. Model downloads occur only when an operator explicitly requests them through their MLX workflow.
- SQLite metadata and filesystem artifacts live under `~/.llm-feedback-finetuning-pipeline` by default; override with `LLM_FEEDBACK_FINETUNING_PIPELINE_HOME`.
- Bind the API to localhost; exposing it to a network requires operator-managed access controls and an API key.
- Dataset deletion does not erase derived adapters; removing an example from an already trained model requires retraining. This is not machine unlearning.
- The dashboard covers overview, correction review/redaction, datasets, local training jobs, paired evaluation with example inspection, registry promotion/rollback, and a prediction playground. Cost assumptions can be calculated through the local API; no measured power or hardware amortization is bundled.
- Synthetic examples only exercise the workflow and cannot indicate business performance.

## Development

```bash
uv sync --extra dev
uv run pytest
uv run ruff check src tests
uv run mypy src/llm_feedback_finetuning_pipeline
cd web && pnpm install && pnpm test && pnpm run lint && pnpm run build
```

The same checks are available through `make test`, `make lint`, `make typecheck`, and `make web-build`. See [CONTRIBUTING.md](CONTRIBUTING.md), [SECURITY.md](.github/SECURITY.md), and [LICENSE](LICENSE).
