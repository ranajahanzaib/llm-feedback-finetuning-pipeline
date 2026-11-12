from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path(
        os.getenv("LLM_FEEDBACK_FINETUNING_PIPELINE_HOME", "~/.llm-feedback-finetuning-pipeline")
    ).expanduser()
    database_url: str | None = os.getenv("LLM_FEEDBACK_FINETUNING_PIPELINE_DATABASE")
    api_key: str | None = os.getenv("LLM_FEEDBACK_FINETUNING_PIPELINE_API_KEY")
    model_id: str = os.getenv("LLM_FEEDBACK_FINETUNING_PIPELINE_MODEL", "Qwen/Qwen3-0.6B-MLX-bf16")
    allow_simulated: bool = os.getenv("LLM_FEEDBACK_FINETUNING_PIPELINE_MODE", "simulated") == "simulated"

    @property
    def database_path(self) -> Path:
        if self.database_url:
            return Path(self.database_url)
        return self.data_dir / "llm-feedback-finetuning-pipeline.sqlite3"

    @property
    def artifacts(self) -> Path:
        return self.data_dir / "artifacts"

    def initialize(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts.mkdir(parents=True, exist_ok=True)


settings = Settings()
