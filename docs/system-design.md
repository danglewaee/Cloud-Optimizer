# System Design Roadmap (Level C)

## Phase 1: Credible Optimizer (done/active)

- [x] LSTM predictor
- [x] MILP optimizer with spot-risk and pod packing
- [x] What-if scenario API
- [x] Baseline-vs-MILP benchmark harness

## Phase 2: Real Infra Integration

- [x] Kubernetes patch generator (`Deployment` + VPA patch)
- [x] Terraform plan patch generator (instance type/count snippet)
- [ ] Safe rollout controller (canary + rollback policy)

## Phase 3: Scale and Reliability

- [x] 200 services / 500 nodes simulation profile
- [ ] 100k+ metrics/min load profile + throughput report
- [ ] Decision latency SLO (`p95 < 3s`)

## Phase 4: Open-Source Packaging

- [ ] Split modules: `autopilot-core`, `autopilot-agent`, `autopilot-dashboard`
- [ ] Helm chart install path
- [ ] Demo scenario and benchmark report in CI

## Suggested KPIs

- Cost reduction vs baseline: `>= 20%`
- Latency constraint violation rate: `< 1%`
- Spot interruption safety: no SLO break under simulated interruption profile
- Optimizer solve time: `p95 < 1s` for 200 services / 500 nodes scenario
