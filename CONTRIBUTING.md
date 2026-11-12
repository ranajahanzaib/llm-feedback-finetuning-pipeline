# Contributing

Contributions to LLM Feedback Fine-Tuning Pipeline are welcome. For substantial changes, open an issue first to discuss scope and compatibility. Keep changes focused on the correction review, dataset governance, local adaptation, evaluation, or deployment workflow.

All community participation follows the [Code of Conduct](CODE_OF_CONDUCT.md).

## Development setup

Requirements: Python 3.11–3.13, [uv](https://docs.astral.sh/uv/), Node.js 20 or newer, and pnpm.

```bash
uv sync --extra dev
cd web && pnpm install
```

Apple Silicon and the optional `mlx` extra are required for real model inference and training. The regular test suite does not download models or require MLX hardware.

## Checks

Run the relevant checks before opening a pull request:

```bash
uv run ruff check src tests
uv run mypy src/llm_feedback_finetuning_pipeline
uv run pytest
cd web && pnpm test && pnpm run lint && pnpm run build
```

Changes to feedback, dataset, evaluation, or registry behavior should include regression tests. Keep model-quality claims tied to reproducible evaluation evidence; synthetic demo results are not production evidence.

## Pull requests

Include the user-visible behavior changed, relevant test results, and any compatibility or privacy implications. Update documentation when commands, configuration, API behavior, or limitations change. Do not include private customer data, credentials, model weights, or generated artifacts.
