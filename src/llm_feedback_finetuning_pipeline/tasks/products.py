from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ProductAttributes(BaseModel):
    """Unknown or unsupported attributes must be null; fields are never inferred."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    category: str | None = Field(default=None, max_length=80)
    gender: str | None = Field(default=None, pattern="^(men|women|unisex|boys|girls)$")
    color: str | None = Field(default=None, max_length=80)
    size: str | None = Field(default=None, max_length=40)
    material: str | None = Field(default=None, max_length=100)
    waterproof: bool | None = None

    @field_validator("category", "color", "size", "material")
    @classmethod
    def empty_to_none(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        if cleaned.casefold() in {"", "unknown", "n/a", "na", "none", "null", "unspecified", "not specified"}:
            return None
        return cleaned


class ProductExtractionTask:
    task_id = "product-attributes"
    schema_version = "1.1.0"
    prompt_version = "product-json-v2"
    system_prompt = (
        "Extract only product attributes explicitly stated in the product text; never guess. "
        "Return exactly one JSON object with these keys: category, gender, color, size, material, waterproof. "
        "Use strings for known text attributes, true or false for waterproof, and null for every unknown field. "
        "Gender must be men, women, unisex, boys, or girls. Keep alpha sizes uppercase (for example XL). "
        "Do not add markdown fences or explanatory text."
    )
    output_model = ProductAttributes

    @classmethod
    def validate_input(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("input_text must not be empty")
        if len(value) > 20_000:
            raise ValueError("input_text exceeds 20000 characters")
        return value

    @classmethod
    def normalize(cls, value: dict[str, Any]) -> dict[str, Any]:
        result = ProductAttributes.model_validate(value).model_dump()
        for field in ("category", "gender", "color", "size", "material"):
            if isinstance(result[field], str):
                normalized = re.sub(r"\s+", " ", result[field])
                result[field] = normalized.upper() if field == "size" else normalized.casefold()
        if result["gender"]:
            result["gender"] = result["gender"].casefold()
        return result

    @classmethod
    def serialize_example(cls, input_text: str, output: dict[str, Any]) -> dict[str, Any]:
        return {
            "messages": [
                {"role": "system", "content": cls.system_prompt},
                {"role": "user", "content": input_text},
                {
                    "role": "assistant",
                    "content": json.dumps(output, ensure_ascii=False, separators=(",", ":")),
                },
            ]
        }
