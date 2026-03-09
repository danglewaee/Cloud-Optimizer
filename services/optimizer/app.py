from __future__ import annotations

import math
from typing import Any

from fastapi import FastAPI
from ortools.linear_solver import pywraplp
from pydantic import BaseModel, Field

app = FastAPI(title="optimizer")

INSTANCE_CATALOG = [
    {
        "type": "m5.large",
        "capacity_rps": 220.0,
        "hourly_cost": 0.096,
        "cpu_cores": 2.0,
        "mem_gb": 8.0,
        "pod_capacity": 30,
        "spot": False,
        "interrupt_prob": 0.0,
    },
    {
        "type": "m5a.large",
        "capacity_rps": 205.0,
        "hourly_cost": 0.086,
        "cpu_cores": 2.0,
        "mem_gb": 8.0,
        "pod_capacity": 30,
        "spot": False,
        "interrupt_prob": 0.0,
    },
    {
        "type": "c6i.large",
        "capacity_rps": 260.0,
        "hourly_cost": 0.102,
        "cpu_cores": 2.0,
        "mem_gb": 4.0,
        "pod_capacity": 28,
        "spot": False,
        "interrupt_prob": 0.0,
    },
    {
        "type": "spot-gpu-small",
        "capacity_rps": 420.0,
        "hourly_cost": 0.11,
        "cpu_cores": 4.0,
        "mem_gb": 16.0,
        "pod_capacity": 26,
        "spot": True,
        "interrupt_prob": 0.08,
    },
]

INSTANCE_BY_TYPE = {i["type"]: i for i in INSTANCE_CATALOG}


class OptimizeInput(BaseModel):
    service: str
    current_nodes: int = Field(ge=1)
    pred_rps: float = Field(gt=0)
    latency_budget_ms: float = Field(default=200.0, gt=0)
    current_instance_type: str = "m5.large"

    # Scheduling knobs for a more realistic optimization problem.
    headroom_ratio: float = Field(default=1.2, ge=1.0, le=2.0)
    target_utilization_max: float = Field(default=0.8, gt=0.4, le=0.95)
    max_nodes: int = Field(default=200, ge=1, le=2000)
    max_spot_ratio: float = Field(default=0.6, ge=0.0, le=1.0)
    pod_rps_capacity: float = Field(default=45.0, gt=1.0)
    pod_cpu_cores: float = Field(default=0.15, gt=0.01)
    pod_mem_gb: float = Field(default=0.35, gt=0.01)


def _latency_util_cap(latency_budget_ms: float, target_utilization_max: float) -> float:
    # Tighter latency budget lowers safe utilization.
    if latency_budget_ms <= 120:
        return min(target_utilization_max, 0.70)
    if latency_budget_ms <= 160:
        return min(target_utilization_max, 0.75)
    if latency_budget_ms <= 220:
        return min(target_utilization_max, 0.82)
    return min(target_utilization_max, 0.88)


@app.post("/optimize")
def optimize(payload: OptimizeInput) -> dict[str, Any]:
    solver = pywraplp.Solver.CreateSolver("CBC_MIXED_INTEGER_PROGRAMMING")
    if solver is None:
        return {
            "service": payload.service,
            "error": "milp_solver_unavailable",
            "target_nodes": payload.current_nodes,
            "target_instance_type": payload.current_instance_type,
            "estimated_hourly_cost": None,
            "utilization": None,
            "score": None,
        }

    x: dict[str, pywraplp.Variable] = {}
    for inst in INSTANCE_CATALOG:
        x[inst["type"]] = solver.IntVar(0, payload.max_nodes, f"x_{inst['type']}")

    total_nodes = solver.IntVar(1, payload.max_nodes, "total_nodes")
    solver.Add(total_nodes == sum(x[t] for t in x))

    required_rps = payload.pred_rps * payload.headroom_ratio
    util_cap = _latency_util_cap(payload.latency_budget_ms, payload.target_utilization_max)
    effective_required_rps = required_rps / util_cap

    # Throughput and pod packing constraints.
    solver.Add(sum(x[i["type"]] * i["capacity_rps"] for i in INSTANCE_CATALOG) >= effective_required_rps)

    required_pods = int(math.ceil(payload.pred_rps / payload.pod_rps_capacity))
    solver.Add(sum(x[i["type"]] * i["pod_capacity"] for i in INSTANCE_CATALOG) >= required_pods)

    # Resource envelope for pod packing realism.
    solver.Add(
        sum(x[i["type"]] * i["cpu_cores"] for i in INSTANCE_CATALOG)
        >= required_pods * payload.pod_cpu_cores / 0.8
    )
    solver.Add(
        sum(x[i["type"]] * i["mem_gb"] for i in INSTANCE_CATALOG)
        >= required_pods * payload.pod_mem_gb / 0.85
    )

    # Spot node risk guardrail.
    spot_nodes_expr = sum(x[i["type"]] for i in INSTANCE_CATALOG if i["spot"])
    solver.Add(spot_nodes_expr <= payload.max_spot_ratio * total_nodes)

    # Smooth scaling penalty to reduce unnecessary churn.
    scale_up = solver.NumVar(0, payload.max_nodes, "scale_up")
    scale_down = solver.NumVar(0, payload.max_nodes, "scale_down")
    solver.Add(total_nodes - payload.current_nodes == scale_up - scale_down)

    # Objective: minimize node cost + spot interruption risk + churn penalty.
    cost_expr = sum(x[i["type"]] * i["hourly_cost"] for i in INSTANCE_CATALOG)
    risk_expr = sum(x[i["type"]] * i["interrupt_prob"] * 0.05 for i in INSTANCE_CATALOG if i["spot"])
    churn_expr = (scale_up + scale_down) * 0.01
    solver.Minimize(cost_expr + risk_expr + churn_expr)

    status = solver.Solve()
    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        return {
            "service": payload.service,
            "error": "no_feasible_solution",
            "target_nodes": payload.current_nodes,
            "target_instance_type": payload.current_instance_type,
            "estimated_hourly_cost": None,
            "utilization": None,
            "score": None,
        }

    allocation = []
    total_capacity = 0.0
    total_cost = 0.0
    for inst in INSTANCE_CATALOG:
        count = int(round(x[inst["type"]].solution_value()))
        if count <= 0:
            continue
        cap = count * inst["capacity_rps"]
        cst = count * inst["hourly_cost"]
        total_capacity += cap
        total_cost += cst
        allocation.append(
            {
                "instance_type": inst["type"],
                "nodes": count,
                "hourly_cost": round(cst, 4),
                "capacity_rps": round(cap, 2),
                "spot": inst["spot"],
            }
        )

    target_nodes = int(round(total_nodes.solution_value()))
    allocation.sort(key=lambda a: a["nodes"], reverse=True)
    dominant_type = allocation[0]["instance_type"] if allocation else payload.current_instance_type

    utilization = payload.pred_rps / max(total_capacity, 1.0)
    score = solver.Objective().Value()

    return {
        "service": payload.service,
        "target_nodes": target_nodes,
        "target_instance_type": dominant_type,
        "estimated_hourly_cost": round(total_cost, 4),
        "utilization": round(utilization, 4),
        "score": round(score, 4),
        "required_pods": required_pods,
        "allocation": allocation,
        "spot_ratio": round(
            (sum(a["nodes"] for a in allocation if a["spot"]) / target_nodes) if target_nodes else 0.0,
            4,
        ),
        "constraints": {
            "effective_required_rps": round(effective_required_rps, 2),
            "target_utilization_max": util_cap,
            "latency_budget_ms": payload.latency_budget_ms,
            "max_spot_ratio": payload.max_spot_ratio,
        },
    }


@app.get("/instance_catalog")
def instance_catalog() -> dict[str, Any]:
    return {"instances": INSTANCE_CATALOG}


@app.get("/health")
def health() -> dict[str, bool]:
    return {"ok": True}
