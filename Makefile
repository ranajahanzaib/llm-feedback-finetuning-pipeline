.PHONY: setup doctor test lint typecheck web-build demo api web

setup:
	uv sync --extra dev

doctor:
	uv run llm-feedback-finetuning-pipeline doctor

test:
	uv run pytest

lint:
	uv run ruff check src tests

typecheck:
	uv run mypy src/llm_feedback_finetuning_pipeline

web-build:
	cd web && pnpm install --frozen-lockfile && pnpm run build

demo:
	uv run llm-feedback-finetuning-pipeline demo

api:
	uv run llm-feedback-finetuning-pipeline serve

web:
	cd web && pnpm run dev
