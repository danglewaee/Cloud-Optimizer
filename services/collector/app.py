from __future__ import annotations

import json
import os
from datetime import datetime

import asyncpg
from aiokafka import AIOKafkaProducer
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="collector")

DB_DSN = os.getenv("DB_DSN", "postgresql://optimizer:optimizer@localhost:5432/optimizer")
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:19092")
TELEMETRY_TOPIC = os.getenv("TELEMETRY_TOPIC", "telemetry.raw")

pool: asyncpg.Pool | None = None
producer: AIOKafkaProducer | None = None


class MetricEvent(BaseModel):
    ts: datetime
    cluster: str
    service: str
    node_id: str
    cpu_pct: float = Field(ge=0, le=100)
    mem_pct: float = Field(ge=0, le=100)
    gpu_pct: float = Field(ge=0, le=100)
    net_mbps: float = Field(ge=0)
    rps: float = Field(ge=0)
    p95_latency_ms: float = Field(ge=0)


CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS metrics (
  id BIGSERIAL PRIMARY KEY,
  ts TIMESTAMPTZ NOT NULL,
  cluster TEXT NOT NULL,
  service TEXT NOT NULL,
  node_id TEXT NOT NULL,
  cpu_pct DOUBLE PRECISION NOT NULL,
  mem_pct DOUBLE PRECISION NOT NULL,
  gpu_pct DOUBLE PRECISION NOT NULL,
  net_mbps DOUBLE PRECISION NOT NULL,
  rps DOUBLE PRECISION NOT NULL,
  p95_latency_ms DOUBLE PRECISION NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_metrics_service_ts ON metrics(service, ts DESC);
"""

INSERT_SQL = """
INSERT INTO metrics (
  ts, cluster, service, node_id, cpu_pct, mem_pct, gpu_pct, net_mbps, rps, p95_latency_ms
) VALUES (
  $1,$2,$3,$4,$5,$6,$7,$8,$9,$10
)
"""


@app.on_event("startup")
async def startup() -> None:
    global pool, producer
    pool = await asyncpg.create_pool(DB_DSN, min_size=1, max_size=10)
    async with pool.acquire() as conn:
        await conn.execute(CREATE_TABLE_SQL)
    producer = AIOKafkaProducer(bootstrap_servers=KAFKA_BROKER)
    await producer.start()


@app.on_event("shutdown")
async def shutdown() -> None:
    if producer:
        await producer.stop()
    if pool:
        await pool.close()


@app.post("/ingest")
async def ingest(events: list[MetricEvent]) -> dict:
    if not events:
        return {"ingested": 0}

    assert pool is not None
    assert producer is not None

    rows = [
        (
            e.ts,
            e.cluster,
            e.service,
            e.node_id,
            e.cpu_pct,
            e.mem_pct,
            e.gpu_pct,
            e.net_mbps,
            e.rps,
            e.p95_latency_ms,
        )
        for e in events
    ]

    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.executemany(INSERT_SQL, rows)

    for e in events:
        payload = json.dumps(e.model_dump(mode="json")).encode("utf-8")
        await producer.send_and_wait(TELEMETRY_TOPIC, payload)

    return {"ingested": len(events)}


@app.get("/services")
async def services() -> dict:
    assert pool is not None
    async with pool.acquire() as conn:
        rows = await conn.fetch("SELECT DISTINCT service FROM metrics ORDER BY service")
    return {"services": [r["service"] for r in rows]}


@app.get("/latest")
async def latest(service: str) -> dict:
    assert pool is not None
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT ts, cluster, service, node_id, cpu_pct, mem_pct, gpu_pct, net_mbps, rps, p95_latency_ms
            FROM metrics
            WHERE service = $1
            ORDER BY ts DESC
            LIMIT 1
            """,
            service,
        )
    return {"event": dict(row) if row else None}


@app.get("/health")
def health() -> dict:
    return {"ok": True, "kafka": KAFKA_BROKER}