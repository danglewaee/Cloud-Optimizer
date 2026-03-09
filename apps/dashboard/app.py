from __future__ import annotations

import os

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

app = FastAPI(title="dashboard")
templates = Jinja2Templates(directory="templates")
DECISION_URL = os.getenv("DECISION_URL", "http://localhost:8005")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/decisions")
async def decisions(limit: int = 10) -> dict:
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(f"{DECISION_URL}/decision_all", params={"limit": limit})
        resp.raise_for_status()
        return resp.json()


@app.get("/api/what_if")
async def what_if(traffic_multiplier: float = 1.3, limit: int = 10) -> dict:
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(
            f"{DECISION_URL}/what_if_all",
            params={"traffic_multiplier": traffic_multiplier, "limit": limit},
        )
        resp.raise_for_status()
        return resp.json()


@app.get("/health")
def health() -> dict:
    return {"ok": True}
