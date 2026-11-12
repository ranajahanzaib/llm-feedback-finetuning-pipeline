from typing import Any, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

from .products import ProductAttributes, ProductExtractionTask

OutputModel = TypeVar("OutputModel", bound=BaseModel)


@runtime_checkable
class TaskSpec(Protocol[OutputModel]):
    """Small structural contract for task schemas used by LLM Feedback Fine-Tuning Pipeline services."""

    task_id: str
    schema_version: str
    prompt_version: str
    system_prompt: str
    output_model: type[OutputModel]

    @classmethod
    def validate_input(cls, value: str) -> str: ...

    @classmethod
    def normalize(cls, value: dict[str, Any]) -> dict[str, Any]: ...

    @classmethod
    def serialize_example(cls, input_text: str, output: dict[str, Any]) -> dict[str, Any]: ...


__all__ = ["ProductAttributes", "ProductExtractionTask", "TaskSpec"]
