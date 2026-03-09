#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
import urllib.parse
import urllib.request
from dataclasses import dataclass


@dataclass
class Result:
    service: str
    baseline_cost: float
    baseline_latency: float
    baseline_util: float
    milp_cost: float
    milp_latency: float
    milp_util: float


def get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=15) as resp:
        return json.loads(resp.read().decode("utf-8"))


def post_json(url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8"))


def baseline_plan(pred_rps: float, current_nodes: int, util_cap: float = 0.75) -> tuple[int, str, float]:
    instance_type = "m5.large"
    capacity = 220.0
    headroom = 1.2
    nodes = max(1, math.ceil((pred_rps * headroom) / (capacity * util_cap)))
    return nodes, instance_type, nodes * 0.096


def run(args: argparse.Namespace) -> None:
    collector = args.collector_url.rstrip("/")
    predictor = args.predictor_url.rstrip("/")
    optimizer = args.optimizer_url.rstrip("/")
    simulator = args.simulator_url.rstrip("/")

    services_payload = get_json(f"{collector}/services")
    services = services_payload.get("services", [])[: args.limit]
    if not services:
        raise SystemExit("No services found. Ensure telemetry is flowing.")

    results: list[Result] = []
    for service in services:
        forecast = get_json(f"{predictor}/forecast/{urllib.parse.quote(service)}?horizon_min=30")
        latest = get_json(f"{collector}/latest?service={urllib.parse.quote(service)}").get("event") or {}

        pred_rps = float(forecast["pred_rps"]) * args.traffic_multiplier
        base_latency = float(latest.get("p95_latency_ms", 90.0))
        current_nodes = args.current_nodes

        b_nodes, b_type, b_cost = baseline_plan(pred_rps, current_nodes)
        b_sim = post_json(
            f"{simulator}/simulate",
            {
                "nodes": b_nodes,
                "expected_rps": pred_rps,
                "per_node_capacity_rps": 220.0,
                "base_latency_ms": base_latency,
            },
        )

        milp = post_json(
            f"{optimizer}/optimize",
            {
                "service": service,
                "current_nodes": current_nodes,
                "pred_rps": pred_rps,
                "latency_budget_ms": 200.0,
                "current_instance_type": b_type,
            },
        )

        milp_nodes = int(milp.get("target_nodes", current_nodes))
        milp_type = milp.get("target_instance_type", "m5.large")
        cap_map = {"m5.large": 220.0, "m5a.large": 205.0, "c6i.large": 260.0, "spot-gpu-small": 420.0}
        milp_cap = cap_map.get(milp_type, 220.0)
        milp_cost = float(milp.get("estimated_hourly_cost") or (0.096 * milp_nodes))

        m_sim = post_json(
            f"{simulator}/simulate",
            {
                "nodes": milp_nodes,
                "expected_rps": pred_rps,
                "per_node_capacity_rps": milp_cap,
                "base_latency_ms": base_latency,
            },
        )

        results.append(
            Result(
                service=service,
                baseline_cost=b_cost,
                baseline_latency=float(b_sim["predicted_p95_latency_ms"]),
                baseline_util=float(b_sim["predicted_utilization"]),
                milp_cost=milp_cost,
                milp_latency=float(m_sim["predicted_p95_latency_ms"]),
                milp_util=float(m_sim["predicted_utilization"]),
            )
        )

    avg_base_cost = statistics.mean(r.baseline_cost for r in results)
    avg_milp_cost = statistics.mean(r.milp_cost for r in results)
    avg_base_lat = statistics.mean(r.baseline_latency for r in results)
    avg_milp_lat = statistics.mean(r.milp_latency for r in results)
    avg_base_util = statistics.mean(r.baseline_util for r in results)
    avg_milp_util = statistics.mean(r.milp_util for r in results)

    cost_reduction = ((avg_base_cost - avg_milp_cost) / avg_base_cost) * 100 if avg_base_cost else 0.0

    print("=== Baseline vs MILP Benchmark ===")
    print(f"services_tested: {len(results)}")
    print(f"traffic_multiplier: {args.traffic_multiplier:.2f}")
    print(f"baseline_avg_hourly_cost: {avg_base_cost:.4f}")
    print(f"milp_avg_hourly_cost: {avg_milp_cost:.4f}")
    print(f"cost_reduction_pct: {cost_reduction:.2f}")
    print(f"baseline_avg_p95_latency_ms: {avg_base_lat:.2f}")
    print(f"milp_avg_p95_latency_ms: {avg_milp_lat:.2f}")
    print(f"baseline_avg_utilization: {avg_base_util:.3f}")
    print(f"milp_avg_utilization: {avg_milp_util:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark baseline autoscaling vs MILP optimizer")
    parser.add_argument("--collector-url", default="http://localhost:8001")
    parser.add_argument("--predictor-url", default="http://localhost:8002")
    parser.add_argument("--optimizer-url", default="http://localhost:8003")
    parser.add_argument("--simulator-url", default="http://localhost:8004")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--traffic-multiplier", type=float, default=1.0)
    parser.add_argument("--current-nodes", type=int, default=12)
    run(parser.parse_args())
