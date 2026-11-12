from __future__ import annotations

import json
import time
from functools import lru_cache
from threading import RLock
from typing import Any, cast

from .config import settings
from .registry import active_model
from .tasks.products import ProductAttributes, ProductExtractionTask

_INFERENCE_LOCK = RLock()


@lru_cache(maxsize=1)
def _load_mlx_model(model_name: str, adapter_path: str | None) -> tuple[Any, Any]:
    from mlx_lm import load

    kwargs = {"adapter_path": adapter_path} if adapter_path else {}
    return cast(tuple[Any, Any], load(model_name, **kwargs))


def _simulate(text: str) -> dict[str, Any]:
    """Deterministic pattern demo; its outputs are explicitly marked simulated."""
    import re

    lower = text.casefold()

    def found(pattern: str) -> str | None:
        match = re.search(pattern, lower)
        return match.group(1) if match else None

    category = found(r"\b(jacket|shirt|coat|shoe|boot|sneaker|bag|backpack|tent|bottle|hat|pants|trousers)\b")
    gender = found(r"\b(men|women|unisex|boys|girls|men's|women's)\b")
    if gender in {"men's", "men"}:
        gender = "men"
    if gender in {"women's", "women"}:
        gender = "women"
    color = found(r"\b(navy blue|dark green|light blue|black|white|red|blue|green|brown|gray|grey|pink|orange|navy)\b")
    size = found(r"(?<!['’])\b(xxxl|xxl|xl|xs|xxs|s|m|l|\d{1,2})\b")
    material = found(r"\b(cotton|polyester|leather|wool|nylon|canvas|denim|silicone|steel)\b")
    waterproof = (
        False
        if re.search(r"\b(?:non[- ]waterproof|not waterproof)\b", lower)
        else True
        if re.search(r"\bwaterproof\b", lower)
        else None
    )
    return ProductExtractionTask.normalize(
        {
            "category": category,
            "gender": gender,
            "color": color,
            "size": size.upper() if size and size in {"xl", "xxl", "xxxl", "xs", "xxs", "s", "m", "l"} else size,
            "material": material,
            "waterproof": waterproof,
        }
    )


def predict(
    text: str,
    *,
    model_id: str | None = None,
    adapter_path: str | None = None,
    base_model: str | None = None,
    mode: str | None = None,
) -> dict[str, Any]:
    text = ProductExtractionTask.validate_input(text)
    selected = mode or ("simulated" if settings.allow_simulated else "mlx")
    started = time.perf_counter()
    token_usage: dict[str, int] | None = None
    if selected == "simulated":
        output = _simulate(text)
        label = "simulated-pattern-demo"
    elif selected == "mlx":
        try:
            from mlx_lm import generate
            from mlx_lm.generate import make_sampler
        except ImportError as exc:
            message = str(exc)
            if "No module named" in message:
                message = "MLX-LM dependencies are not installed; run `uv sync --extra mlx`. " + message
            else:
                message = "MLX backend could not initialize on this machine: " + message
            raise RuntimeError(message) from exc
        active = active_model()
        model_name = base_model or (active or {}).get("base_model", settings.model_id)
        adapter = adapter_path or ((active or {}).get("adapter_path"))
        adapter_path_resolved = None
        if adapter:
            from .db import path_for

            adapter_path_resolved = str(path_for("models", adapter))
        with _INFERENCE_LOCK:
            model, tokenizer = _load_mlx_model(model_name, adapter_path_resolved)
            prompt = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": ProductExtractionTask.system_prompt},
                    {"role": "user", "content": text},
                ],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            raw = generate(model, tokenizer, prompt=prompt, max_tokens=180, sampler=make_sampler(temp=0.0))
        try:
            parsed = json.loads(raw)
            if not isinstance(parsed, dict) or set(parsed) != set(ProductAttributes.model_fields):
                raise ValueError("response must contain every required field and no extras")
            output = ProductExtractionTask.normalize(parsed)
        except Exception as exc:
            raise ValueError(f"Model returned output that failed raw JSON/schema validation: {exc}") from exc
        token_usage = {"completion_tokens": len(tokenizer.encode(raw))}
        label = model_id or (active or {}).get("id") or model_name
    else:
        raise ValueError("mode must be simulated or mlx")
    result = {
        "task_id": ProductExtractionTask.task_id,
        "model_id": label,
        "output": output,
        "validation": "valid",
        "duration_ms": round((time.perf_counter() - started) * 1000, 2),
        "mode": selected,
        "warnings": ["Deterministic simulated pattern output; not an ML measurement."]
        if selected == "simulated"
        else [],
    }
    if token_usage is not None:
        result["token_usage"] = token_usage
    return result
