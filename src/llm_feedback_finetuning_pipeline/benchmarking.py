"""Transparent local inference cost scenarios; missing assumptions stay unavailable."""

from __future__ import annotations

from typing import Any

REQUEST_SCENARIOS = (1_000, 10_000, 100_000)


def local_cost_scenarios(
    *,
    accuracy: float,
    power_watts: float | None,
    electricity_usd_per_kwh: float | None,
    inference_seconds: float | None,
    training_cost_usd: float | None,
    monthly_hardware_cost_usd: float | None,
    training_amortization_months: int = 12,
) -> dict[str, Any]:
    """Estimate costs from operator-provided assumptions, without fabricating missing values."""
    if not 0 <= accuracy <= 1:
        raise ValueError("accuracy must be between 0 and 1")
    if training_amortization_months < 1:
        raise ValueError("training_amortization_months must be at least 1")
    missing = [
        name
        for name, value in (
            ("power_watts", power_watts),
            ("electricity_usd_per_kwh", electricity_usd_per_kwh),
            ("inference_seconds", inference_seconds),
            ("training_cost_usd", training_cost_usd),
            ("monthly_hardware_cost_usd", monthly_hardware_cost_usd),
        )
        if value is None
    ]
    assumptions = {
        "power_watts": power_watts,
        "power_source": "operator-provided; measurement status not independently verified",
        "electricity_usd_per_kwh": electricity_usd_per_kwh,
        "inference_seconds": inference_seconds,
        "training_cost_usd": training_cost_usd,
        "monthly_hardware_cost_usd": monthly_hardware_cost_usd,
        "training_amortization_months": training_amortization_months,
        "accuracy": accuracy,
    }
    if missing or accuracy == 0:
        return {
            "available": False,
            "reason": "Missing assumptions: " + ", ".join(missing) if missing else "Accuracy is zero",
            "assumptions": assumptions,
            "scenarios": {str(count): None for count in REQUEST_SCENARIOS},
        }
    assert power_watts is not None
    assert electricity_usd_per_kwh is not None
    assert inference_seconds is not None
    assert training_cost_usd is not None
    assert monthly_hardware_cost_usd is not None
    electricity_per_request = power_watts / 1000 * inference_seconds / 3600 * electricity_usd_per_kwh
    training_monthly = training_cost_usd / training_amortization_months
    scenarios = {}
    for count in REQUEST_SCENARIOS:
        monthly = count * electricity_per_request + training_monthly + monthly_hardware_cost_usd
        scenarios[str(count)] = {
            "monthly_requests": count,
            "monthly_cost_usd": round(monthly, 6),
            "cost_per_correct_task_usd": round(monthly / (count * accuracy), 9),
            "inference_electricity_per_request_usd": round(electricity_per_request, 9),
            "training_cost_amortized_monthly_usd": round(training_monthly, 6),
            "hardware_cost_monthly_usd": monthly_hardware_cost_usd,
        }
    return {"available": True, "assumptions": assumptions, "scenarios": scenarios}
