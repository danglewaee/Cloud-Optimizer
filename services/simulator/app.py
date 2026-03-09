from __future__ import annotations

import math

from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="simulator")


class SimulationInput(BaseModel):
    nodes: int = Field(ge=1)
    expected_rps: float = Field(gt=0)
    per_node_capacity_rps: float = Field(gt=0)
    base_latency_ms: float = Field(default=40.0, gt=0)


def erlang_c(c: int, rho: float) -> float:
    if rho >= 1.0:
        return 1.0

    numer = (c * rho) ** c / math.factorial(c) * (1.0 / (1.0 - rho))
    denom = sum(((c * rho) ** k) / math.factorial(k) for k in range(c)) + numer
    return numer / denom


@app.post("/simulate")
def simulate(payload: SimulationInput) -> dict:
    c = payload.nodes
    total_capacity = payload.per_node_capacity_rps * c
    rho = payload.expected_rps / max(total_capacity, 1e-9)

    if rho >= 1:
        return {
            "predicted_p95_latency_ms": 2000.0,
            "predicted_utilization": 1.0,
            "risk": "overloaded",
        }

    wait_prob = erlang_c(c, rho)
    queue_factor = wait_prob / max(c * (1 - rho), 1e-6)
    predicted_p95 = payload.base_latency_ms * (1 + 1.8 * queue_factor + (rho * 0.9))

    risk = "low"
    if rho > 0.85 or predicted_p95 > 180:
        risk = "high"
    elif rho > 0.72 or predicted_p95 > 140:
        risk = "medium"

    return {
        "predicted_p95_latency_ms": round(predicted_p95, 2),
        "predicted_utilization": round(rho, 3),
        "risk": risk,
    }


@app.get("/health")
def health() -> dict:
    return {"ok": True}