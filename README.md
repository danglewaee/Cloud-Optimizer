# Self-Optimizing Cloud Infrastructure AI 

MVP system that simulates cloud telemetry, predicts workload, recommends scaling actions, and visualizes cost/latency impact.

## What is included

- `telemetry-simulator`: generates metrics for services and nodes.
- `collector`: ingests telemetry, writes to TimescaleDB, publishes to stream.
- `predictor`: LSTM forecasting service with heuristic fallback.
- `optimizer`: MILP scheduler (OR-Tools) with pod packing and spot-risk guardrails.
- `simulator`: queueing-based latency estimator.
- `decision`: predictive hybrid HPA+VPA orchestration with risk-aware downscale guardrail and patch artifacts.
- `dashboard`: real-time UI with what-if analysis.

## Architecture

`telemetry-simulator -> collector -> (timescaledb + kafka) -> predictor/optimizer/simulator -> decision -> dashboard`

## Quick start

Requirements:

- Docker + Docker Compose

Run default profile:

```bash
make up
```

Run scale profile (200 services / 500 nodes):

```bash
make up-scale
```

Open:

- Dashboard: `http://localhost:8080`
- Decision API: `http://localhost:8005/docs`
- Predictor API: `http://localhost:8002/docs`
- Optimizer API: `http://localhost:8003/docs`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000`

Stop:

```bash
make down
```

## Key API examples

- Ingest telemetry: `POST /ingest` on collector
- Forecast one service: `GET /forecast/{service}?horizon_min=30`
- Optimize plan (MILP): `POST /optimize`
- Simulate latency: `POST /simulate`
- Build recommendation: `GET /decision/{service}`
- List recommendations: `GET /decision_all`
- What-if (single service): `GET /what_if/{service}?traffic_multiplier=1.3`
- What-if (all services): `GET /what_if_all?traffic_multiplier=1.3&limit=10`
- Generate infra patches (Deployment + VPA + Terraform): `GET /artifacts/{service}?traffic_multiplier=1.2`

## Train LSTM predictor

After the stack is running and metrics are flowing:

```bash
docker compose -f infra/docker-compose/docker-compose.yml exec predictor python train_lstm.py --dsn postgresql://optimizer:optimizer@postgres:5432/optimizer --output /app/checkpoints/rps_lstm.pt --epochs 15
```

Restart predictor to load the new model:

```bash
docker compose -f infra/docker-compose/docker-compose.yml restart predictor
```

Check model status:

```bash
curl http://localhost:8002/model/status
```

## Benchmark baseline vs MILP

```bash
make bench
```

Or customize:

```bash
python scripts/benchmark_baseline_vs_milp.py --limit 50 --traffic-multiplier 1.5
```

## Design docs

- `docs/architecture.md`
- `docs/system-design.md`

## Bench hooks to implement next

- Add true pipeline throughput benchmark for 100k+ metrics/min.
- Add autonomous apply mode with rollout guardrails.
- Add multi-cluster control plane mode (`autopilot-core` + `autopilot-agent`).
