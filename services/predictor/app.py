from __future__ import annotations

import os
from pathlib import Path

import asyncpg
import numpy as np
import torch
from fastapi import FastAPI
from pydantic import BaseModel

from data import apply_norm, undo_norm
from model import LSTMForecaster, load_checkpoint

app = FastAPI(title="predictor")
DB_DSN = os.getenv("DB_DSN", "postgresql://optimizer:optimizer@localhost:5432/optimizer")
MODEL_PATH = os.getenv("MODEL_PATH", "/app/checkpoints/rps_lstm.pt")

pool: asyncpg.Pool | None = None
model: LSTMForecaster | None = None
ckpt_meta: dict | None = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ForecastResponse(BaseModel):
    service: str
    horizon_min: int
    pred_rps: float
    pred_cpu_pct: float
    confidence: float
    model_source: str


@app.on_event("startup")
async def startup() -> None:
    global pool, model, ckpt_meta
    pool = await asyncpg.create_pool(DB_DSN, min_size=1, max_size=5)

    if Path(MODEL_PATH).exists():
        model = LSTMForecaster(input_size=2, hidden_size=64, num_layers=2, dropout=0.1).to(device)
        ckpt_meta = load_checkpoint(model, MODEL_PATH, device)
    else:
        model = None
        ckpt_meta = None


@app.on_event("shutdown")
async def shutdown() -> None:
    if pool:
        await pool.close()


async def get_recent_series(service: str, limit: int = 240) -> tuple[list[float], list[float]]:
    assert pool is not None
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT rps, cpu_pct
            FROM metrics
            WHERE service = $1
            ORDER BY ts DESC
            LIMIT $2
            """,
            service,
            limit,
        )

    rps_series = [float(r["rps"]) for r in rows]
    cpu_series = [float(r["cpu_pct"]) for r in rows]
    return rps_series, cpu_series


def heuristic_forecast(service: str, horizon_min: int, rps_series: list[float], cpu_series: list[float]) -> ForecastResponse:
    if not rps_series:
        return ForecastResponse(
            service=service,
            horizon_min=horizon_min,
            pred_rps=120.0,
            pred_cpu_pct=55.0,
            confidence=0.35,
            model_source="heuristic",
        )

    n = len(rps_series)
    last = rps_series[0]
    avg = sum(rps_series) / n
    trend = 0.0
    if n > 8:
        newer = sum(rps_series[:4]) / 4
        older = sum(rps_series[4:8]) / 4
        trend = newer - older

    horizon_factor = max(1.0, horizon_min / 5.0)
    pred_rps = max(1.0, avg * 0.55 + last * 0.45 + trend * 0.15 * horizon_factor)
    pred_cpu = max(5.0, min(99.0, (sum(cpu_series[: min(12, n)]) / min(12, n)) * (pred_rps / max(last, 1.0))))
    confidence = min(0.9, 0.35 + min(n, 120) / 300)

    return ForecastResponse(
        service=service,
        horizon_min=horizon_min,
        pred_rps=round(pred_rps, 2),
        pred_cpu_pct=round(pred_cpu, 2),
        confidence=round(confidence, 2),
        model_source="heuristic",
    )


def lstm_forecast(service: str, horizon_min: int, rps_series: list[float], cpu_series: list[float]) -> ForecastResponse | None:
    if model is None or ckpt_meta is None:
        return None

    window_size = int(ckpt_meta.get("window_size", 24))
    if len(rps_series) < window_size:
        return None

    mean = np.array(ckpt_meta["mean"], dtype=np.float32)
    std = np.array(ckpt_meta["std"], dtype=np.float32)

    values_desc = np.column_stack((np.array(rps_series, dtype=np.float32), np.array(cpu_series, dtype=np.float32)))
    values_asc = values_desc[::-1]

    window = values_asc[-window_size:]
    current = window.copy()

    steps = max(1, int(round(horizon_min / 5.0)))
    for _ in range(steps):
        x = apply_norm(current, mean, std)
        x_tensor = torch.from_numpy(x).unsqueeze(0).to(device)
        with torch.no_grad():
            pred_norm = model(x_tensor).cpu().numpy()[0]
        pred = undo_norm(pred_norm, mean, std)
        pred[0] = max(1.0, pred[0])
        pred[1] = np.clip(pred[1], 1.0, 99.0)
        current = np.vstack([current[1:], pred])

    final_pred = current[-1]
    confidence = float(np.clip(0.6 + min(len(rps_series), 300) / 1000.0, 0.6, 0.92))

    return ForecastResponse(
        service=service,
        horizon_min=horizon_min,
        pred_rps=round(float(final_pred[0]), 2),
        pred_cpu_pct=round(float(final_pred[1]), 2),
        confidence=round(confidence, 2),
        model_source="lstm",
    )


@app.get("/forecast/{service}", response_model=ForecastResponse)
async def forecast(service: str, horizon_min: int = 30) -> ForecastResponse:
    rps_series, cpu_series = await get_recent_series(service)

    model_resp = lstm_forecast(service, horizon_min, rps_series, cpu_series)
    if model_resp is not None:
        return model_resp

    return heuristic_forecast(service, horizon_min, rps_series, cpu_series)


@app.get("/model/status")
def model_status() -> dict:
    return {
        "loaded": model is not None and ckpt_meta is not None,
        "model_path": MODEL_PATH,
        "window_size": ckpt_meta.get("window_size") if ckpt_meta else None,
        "val_loss": ckpt_meta.get("val_loss") if ckpt_meta else None,
    }


@app.get("/health")
def health() -> dict:
    return {"ok": True}

