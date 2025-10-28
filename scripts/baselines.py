#!/usr/bin/env python3
# scripts/baselines.py
"""
Compute simple time-based baselines for NDVI or similar targets.
Now uses robust temporal splits and NaN-safe evaluation.
"""

import argparse, pandas as pd, json, numpy as np
from pathlib import Path

# === Baseline Methods ===================================================== #

def seasonal_mean_baseline(df, target_col):
    """Mean per calendar month from train, applied to val/test."""
    train = df[df["split"] == "train"].copy()
    ref = train.groupby(df["date"].dt.month)[target_col].mean()
    def predict(d):
        return d["date"].dt.month.map(ref)
    return predict

def last_month_persistence(df, target_col):
    """Predict current value = last month's value."""
    df = df.sort_values("date").copy()
    def predict(d):
        s = d.set_index("date")[target_col].shift(1)
        return s.reindex(d["date"]).values
    return predict

# === Evaluation Helper ==================================================== #

def evaluate(df, yhat, target_col):
    """Compute R², MAE, RMSE with NaN-safe handling."""
    y = df[target_col].values
    yhat = np.array(yhat, dtype=float)
    mask = ~np.isnan(yhat) & ~np.isnan(y)
    if mask.sum() == 0:
        return {"R2": float("nan"), "MAE": float("nan"), "RMSE": float("nan"), "n": 0}

    y, yhat = y[mask], yhat[mask]
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    mae = float(np.mean(np.abs(y - yhat)))
    rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))
    return {"R2": r2, "MAE": mae, "RMSE": rmse, "n": int(mask.sum())}

# === Main ================================================================= #

def main():
    ap = argparse.ArgumentParser(description="Compute simple NDVI baselines")
    ap.add_argument("--region", required=True, help="Region folder name under data/")
    ap.add_argument("--target", default="ndvi_delta", help="Target column name")
    args = ap.parse_args()

    ddir = Path(f"data/{args.region}")
    infile = ddir / "training_window.csv"
    if not infile.exists():
        raise FileNotFoundError(f"Missing training window file: {infile}")

    df = pd.read_csv(infile, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # === Robust temporal split (always produces train/val/test) === #
    if "split" not in df.columns:
        n = len(df)
        df["split"] = "train"
        df.loc[int(0.6 * n):int(0.8 * n), "split"] = "val"
        df.loc[int(0.8 * n):, "split"] = "test"

    # === Run all baselines === #
    results = {}
    baselines = {
        "seasonal_mean": seasonal_mean_baseline,
        "last_month_persistence": last_month_persistence,
    }

    for name, maker in baselines.items():
        pred_fn = maker(df, args.target)
        out = {}
        for split in ["val", "test"]:
            d = df[df["split"] == split].copy()
            yhat = pred_fn(d)
            out[split] = evaluate(d, yhat, args.target)
        results[name] = out

    # === Save results === #
    out_dir = Path(f"models/{args.region}")
    out_dir.mkdir(parents=True, exist_ok=True)
    outfile = out_dir / f"baseline_metrics_{args.target}.json"
    outfile.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))

# ========================================================================== #

if __name__ == "__main__":
    main()
