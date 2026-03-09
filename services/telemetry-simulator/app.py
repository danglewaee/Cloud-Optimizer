from __future__ import annotations

import asyncio
import os
import random
from datetime import datetime, timezone

import httpx
from fastapi import FastAPI
from prometheus_client import Counter, Gauge, generate_latest
from starlette.responses import PlainTextResponse

app = FastAPI(title="telemetry-simulator")

collector_url = os.getenv("COLLECTOR_URL", "http://localhost:8001")
interval_sec = int(os.getenv("SIM_INTERVAL_SEC", "2"))
service_count = int(os.getenv("SIM_SERVICE_COUNT", "20"))
node_count = int(os.getenv("SIM_NODE_COUNT", "50"))

sent_events = Counter("sim_sent_events_total", "Total generated events")
last_batch_size = Gauge("sim_last_batch_size", "Last emitted batch size")

_task: asyncio.Task | None = None


def generate_event(service_name: str, node_id: str) -> dict:
    cpu = max(5.0, min(99.0, random.gauss(58, 16)))
    mem = max(10.0, min(99.0, random.gauss(65, 12)))
    gpu = max(0.0, min(99.0, random.gauss(28, 20)))
    net = max(1.0, random.gauss(160, 45))
    rps = max(1.0, random.gauss(120, 45))
    latency = max(10.0, 70 + (cpu * 0.9) + random.gauss(0, 8))

    return {
        "ts": datetime.now(timezone.utc).isoformat(),
        "cluster": "sim-cluster-a",
        "service": service_name,
        "node_id": node_id,
        "cpu_pct": round(cpu, 2),
        "mem_pct": round(mem, 2),
        "gpu_pct": round(gpu, 2),
        "net_mbps": round(net, 2),
        "rps": round(rps, 2),
        "p95_latency_ms": round(latency, 2),
    }


async def sender_loop() -> None:
    services = [f"service-{i:02d}" for i in range(1, service_count + 1)]
    nodes = [f"node-{i:03d}" for i in range(1, node_count + 1)]

    async with httpx.AsyncClient(timeout=10.0) as client:
        while True:
            events = []
            for service_name in services:
                sampled_nodes = random.sample(nodes, k=min(3, len(nodes)))
                for node_id in sampled_nodes:
                    events.append(generate_event(service_name, node_id))

            try:
                await client.post(f"{collector_url}/ingest", json=events)
                sent_events.inc(len(events))
                last_batch_size.set(len(events))
            except Exception:
                pass

            await asyncio.sleep(interval_sec)


@app.on_event("startup")
async def startup() -> None:
    global _task
    _task = asyncio.create_task(sender_loop())


@app.on_event("shutdown")
async def shutdown() -> None:
    global _task
    if _task:
        _task.cancel()


@app.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "interval_sec": interval_sec,
        "service_count": service_count,
        "node_count": node_count,
    }


@app.get("/metrics")
def metrics() -> PlainTextResponse:
    return PlainTextResponse(generate_latest().decode("utf-8"))