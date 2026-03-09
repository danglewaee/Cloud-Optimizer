from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import asyncpg
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from data import build_sequences, normalize_features
from model import LSTMForecaster


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LSTM predictor on metrics table")
    parser.add_argument("--dsn", required=True, help="Postgres DSN")
    parser.add_argument("--output", default="checkpoints/rps_lstm.pt", help="Model checkpoint path")
    parser.add_argument("--window-size", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-rows", type=int, default=120000)
    return parser.parse_args()


async def fetch_rows(dsn: str, max_rows: int) -> np.ndarray:
    conn = await asyncpg.connect(dsn)
    try:
        rows = await conn.fetch(
            """
            SELECT rps, cpu_pct
            FROM metrics
            ORDER BY ts ASC
            LIMIT $1
            """,
            max_rows,
        )
    finally:
        await conn.close()

    if not rows:
        return np.empty((0, 2), dtype=np.float32)

    return np.array([[float(r["rps"]), float(r["cpu_pct"])] for r in rows], dtype=np.float32)


def train(features: np.ndarray, output_path: str, window_size: int, epochs: int, batch_size: int, lr: float) -> None:
    normed, mean, std = normalize_features(features)
    x, y = build_sequences(normed, window_size=window_size, horizon_steps=1)
    if len(x) < 200:
        raise RuntimeError("Not enough sequence samples for training. Need at least 200 windows.")

    split = int(len(x) * 0.85)
    x_train, y_train = x[:split], y[:split]
    x_val, y_val = x[split:], y[split:]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LSTMForecaster(input_size=2, hidden_size=64, num_layers=2, dropout=0.1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=batch_size,
        shuffle=True,
    )
    val_x = torch.from_numpy(x_val).to(device)
    val_y = torch.from_numpy(y_val).to(device)

    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(xb)

        model.eval()
        with torch.no_grad():
            val_pred = model(val_x)
            val_loss = loss_fn(val_pred, val_y).item()

        train_loss = total_loss / len(x_train)
        print(f"epoch={epoch:02d} train_loss={train_loss:.5f} val_loss={val_loss:.5f}")

        if val_loss < best_val:
            best_val = val_loss
            ckpt = {
                "state_dict": model.state_dict(),
                "window_size": window_size,
                "input_size": 2,
                "mean": mean.tolist(),
                "std": std.tolist(),
                "val_loss": best_val,
            }
            output = Path(output_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(ckpt, output)

    print(f"saved_checkpoint={output_path} best_val_loss={best_val:.5f}")


async def main() -> None:
    args = parse_args()
    data = await fetch_rows(args.dsn, args.max_rows)
    if len(data) == 0:
        raise RuntimeError("No data found in metrics table")
    train(
        features=data,
        output_path=args.output,
        window_size=args.window_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )


if __name__ == "__main__":
    asyncio.run(main())

