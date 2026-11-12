from __future__ import annotations

import json
import random
from typing import Any

from .db import new_id, now, transaction
from .tasks.products import ProductExtractionTask

COLORS = ["navy blue", "black", "white", "red", "forest green", "gray", "sand", "blue"]
MATERIALS = ["cotton", "polyester", "leather", "wool", "nylon", "canvas", None]
CATEGORIES = ["jacket", "shirt", "coat", "sneaker", "boot", "backpack", "hat", "pants"]
GENDERS = ["men", "women", "unisex", None]
SIZES = ["S", "M", "L", "XL", "8", "10", None]


def seed_demo(count: int = 120, seed: int = 2026) -> dict[str, Any]:
    rng = random.Random(seed)
    records = []
    for _index in range(count):
        category, color, material = (
            rng.choice(CATEGORIES),
            rng.choice(COLORS),
            rng.choice(MATERIALS),
        )
        gender, size = rng.choice(GENDERS), rng.choice(SIZES)
        waterproof = rng.choice([True, False, None])
        tokens = [
            f"{gender}'s" if gender else "",
            f"{color}",
            f"{category}",
            f"size {size}" if size else "",
            f"{material} shell" if material else "",
            "waterproof" if waterproof else "",
            "" if waterproof is not False else "non-waterproof",
        ]
        rng.shuffle(tokens)
        text = " ".join(token for token in tokens if token).strip().capitalize() + "."
        output = ProductExtractionTask.normalize(
            {
                "category": category,
                "gender": gender,
                "color": color,
                "size": size,
                "material": material,
                "waterproof": waterproof,
            }
        )
        records.append((new_id("fb"), text, json.dumps(output), now()))
    with transaction() as con:
        if con.execute("SELECT 1 FROM feedback LIMIT 1").fetchone():
            return {
                "loaded": 0,
                "message": "Feedback already exists; synthetic fixture was not duplicated.",
            }
        for feedback_id, text, label_json, created in records:
            con.execute(
                "INSERT INTO feedback(id,task_id,schema_version,input_text,corrected_output,source,"
                "source_record_id,source_group_id,reviewer,status,notes,privacy_flags,created_at,reviewed_at) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    feedback_id,
                    "product-attributes",
                    ProductExtractionTask.schema_version,
                    text,
                    label_json,
                    "synthetic-demo",
                    f"synthetic-{feedback_id}",
                    feedback_id,
                    "synthetic-fixture",
                    "approved",
                    "Synthetic fixture, pre-approved for demo only",
                    "[]",
                    created,
                    created,
                ),
            )
    return {
        "loaded": count,
        "status": "synthetic demo data; not customer data or a quality claim",
        "seed": seed,
    }
