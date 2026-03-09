# Architecture

## Services

- `telemetry-simulator`: generates service/node telemetry.
- `collector`: ingests telemetry, persists to TimescaleDB, emits stream events.
- `predictor`: LSTM forecasting (`rps`, `cpu`) with heuristic fallback.
- `optimizer`: MILP-based scheduler with cost, latency, pod packing, and spot-risk constraints.
- `simulator`: queueing-style latency risk simulation.
- `decision`: predictive hybrid HPA+VPA orchestrator with risk guardrails and patch artifacts.
- `dashboard`: realtime and what-if UI.

## Optimization Flow

1. Predictor forecasts demand for horizons 5m, 15m, and 60m.
2. Decision layer computes a planning demand and calls MILP optimizer.
3. Optimizer solves node mix with throughput, latency-utilization, pod packing, and spot ratio constraints.
4. Simulator validates expected latency and risk.
5. Guardrail simulation tests +20% surge before downscale; if unsafe, downscale is blocked.
6. Decision API returns final recommendation + impact + infrastructure patches.

## Current Constraints Model

- Throughput: `sum(nodes_i * capacity_rps_i) >= required_rps / util_cap`
- Pod packing: `sum(nodes_i * pod_capacity_i) >= required_pods`
- CPU envelope: `sum(nodes_i * cpu_i) >= required_pods * pod_cpu / 0.8`
- Memory envelope: `sum(nodes_i * mem_i) >= required_pods * pod_mem / 0.85`
- Spot ratio: `spot_nodes <= max_spot_ratio * total_nodes`

## Decision Outputs

- Horizontal decision: `recommended.nodes`, `recommended.instance_type`
- Vertical decision: `recommended.vertical.cpu_request_m`, `recommended.vertical.memory_request_mi`
- Guardrail status: `policy.downscale_guardrail_triggered`, `policy.guardrail_reason`
- Artifacts: Kubernetes deployment patch, VPA patch, Terraform module snippet

## Level C Upgrade Path

- Replace static catalog with cloud pricing adapters.
- Add true 100k+ metrics/min throughput benchmark report.
- Add autonomous apply mode with rollout guardrails.
- Add multi-cluster control plane (`autopilot-core` + `autopilot-agent`).
