from __future__ import annotations

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
#!/usr/bin/env python3
"""Unified entry point for the regional climate insight pipeline."""

import argparse
import csv
import importlib
import json
import os
import statistics
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Dependency helper
# ---------------------------------------------------------------------------


def require(pkg: str, import_name: Optional[str] = None):
    """Import *pkg*, installing it on-demand when permitted."""

    module_name = import_name or pkg
    try:
        return importlib.import_module(module_name)
    except ImportError:
        if os.environ.get("OFFLINE_MODE") == "1":
            print(f"⚠️ Offline mode: '{pkg}' not installed, skipping auto-install.")
            return None

        print(f"📦 Installing missing dependency: {pkg}")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
        except subprocess.CalledProcessError as exc:
            print(f"❌ Failed to install '{pkg}': {exc}")
            return None
        _log_auto_install(pkg)
        return importlib.import_module(module_name)


_RUNTIME_LOG = Path(__file__).resolve().parents[1] / ".runtime_log.txt"


def _log_auto_install(pkg: str) -> None:
    try:
        timestamp = datetime.utcnow().isoformat()
        with _RUNTIME_LOG.open("a", encoding="utf-8") as handle:
            handle.write(f"{timestamp}\t{pkg}\n")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Pipeline orchestration helpers
# ---------------------------------------------------------------------------

SCRIPTS_DIR = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[1]
_CONFIG_MODULE = None

SRC_DIR = ROOT / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _config():
    global _CONFIG_MODULE
    if _CONFIG_MODULE is None:
        from src.regional_agent import config as cfg  # noqa: WPS433 - runtime import

        _CONFIG_MODULE = cfg
    return _CONFIG_MODULE

# The core libraries required by downstream scripts.  The tuple structure is
# (package-name, import-name).
ESSENTIAL_PACKAGES: Tuple[Tuple[str, Optional[str]], ...] = (
    ("pyyaml", "yaml"),
    ("pandas", None),
    ("numpy", None),
    ("requests", None),
    ("folium", None),
    ("matplotlib", None),
    ("scikit-learn", None),
    ("joblib", None),
)


@dataclass
class StepConfig:
    name: str
    command: Optional[Sequence[str]] = None
    allow_failure: bool = False
    skip: bool = False
    callable: Optional[callable] = None  # type: ignore[assignment]


def bootstrap_dependencies(packages: Iterable[Tuple[str, Optional[str]]]) -> list[str]:
    missing: list[str] = []
    for pkg, import_name in packages:
        module = require(pkg, import_name)
        if module is None:
            missing.append(pkg)
    if missing:
        if os.environ.get("OFFLINE_MODE") == "1":
            print(
                "⚠️ Some dependencies could not be installed in offline mode: "
                + ", ".join(missing)
            )
        else:
            raise RuntimeError(
                "Unable to import required packages: " + ", ".join(missing)
            )
    else:
        print("✅ All pipeline dependencies available.")
    return missing


def build_command(script: str, *args: str) -> Sequence[str]:
    script_path = SCRIPTS_DIR / script
    if not script_path.exists():
        raise FileNotFoundError(f"Pipeline script missing: {script_path}")
    return (sys.executable, str(script_path), *args)


def run_step(step: StepConfig) -> None:
    if step.skip:
        print(f"⏭️  Skipping {step.name}")
        return

    print(f"\n🚀 {step.name}")
    if step.callable is not None:
        try:
            step.callable()
        except Exception as exc:
            if step.allow_failure:
                print(f"⚠️ {step.name} failed but is allowed to continue: {exc}")
            else:
                raise
        else:
            print(f"✅ {step.name} complete.")
        return

    if step.command is None:
        raise ValueError(f"Step '{step.name}' has neither command nor callable.")

    print("   " + " ".join(step.command))
    try:
        subprocess.check_call(step.command, cwd=str(ROOT))
        print(f"✅ {step.name} complete.")
    except subprocess.CalledProcessError as exc:
        if step.allow_failure:
            print(f"⚠️ {step.name} failed but is allowed to continue: {exc}")
        else:
            raise


# ---------------------------------------------------------------------------
# Region setup helpers
# ---------------------------------------------------------------------------


def _ensure_region_profile(region: str) -> Path:
    try:
        return _config().resolve_region_config_path(region)
    except FileNotFoundError:
        defaults = Path("regions/profiles/insight.defaults.yml")
        if not defaults.exists():
            defaults = Path("config/insight.defaults.yml")
        if not defaults.exists():
            raise
        target = Path("regions/profiles") / f"insight.{region}.yml"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(defaults.read_text())
        print(f"🆕 Created region profile from defaults → {target}")
        return target


def _seed_sample_fetches(region: str) -> None:
    """Create lightweight synthetic fetch outputs for offline runs."""

    data_root = ROOT / "data" / region
    current_dir = data_root / "current"
    data_root.mkdir(parents=True, exist_ok=True)
    current_dir.mkdir(parents=True, exist_ok=True)
    targets = [data_root, current_dir]

    start = datetime(2023, 1, 1)
    periods = 180

    def random_series(mean: float, scale: float) -> list[float]:
        import random

        return [random.gauss(mean, scale) for _ in range(periods)]

    dates = [(start + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(periods)]
    precip_values = [max(0.0, value) for value in random_series(4.0, 3.0)]
    soil_surface = [min(0.5, max(0.05, value)) for value in random_series(0.22, 0.03)]
    soil_root = [min(0.6, max(0.05, value)) for value in random_series(0.28, 0.04)]
    ndvi_values = [min(0.9, max(0.1, value)) for value in random_series(0.65, 0.05)]
    ndvi_mean = sum(ndvi_values) / len(ndvi_values)
    ndvi_anomaly = [value - ndvi_mean for value in ndvi_values]
    t2m_mean = random_series(24.0, 3.0)
    t2m_max = random_series(29.0, 3.0)
    t2m_min = random_series(19.0, 3.0)

    datasets = {
        "chirps_gee.csv": (
            ["date", "precip_mm_sum"],
            [{"date": d, "precip_mm_sum": round(v, 3)} for d, v in zip(dates, precip_values)],
        ),
        "soil_gee.csv": (
            ["date", "soil_surface_moisture", "soil_rootzone_moisture"],
            [
                {
                    "date": d,
                    "soil_surface_moisture": round(s, 3),
                    "soil_rootzone_moisture": round(r, 3),
                }
                for d, s, r in zip(dates, soil_surface, soil_root)
            ],
        ),
        "ndvi_gee.csv": (
            ["date", "ndvi", "ndvi_anomaly"],
            [
                {"date": d, "ndvi": round(v, 3), "ndvi_anomaly": round(a, 3)}
                for d, v, a in zip(dates, ndvi_values, ndvi_anomaly)
            ],
        ),
        "openmeteo.csv": (
            ["date", "t2m_mean", "t2m_max", "t2m_min"],
            [
                {
                    "date": d,
                    "t2m_mean": round(m, 2),
                    "t2m_max": round(x, 2),
                    "t2m_min": round(n, 2),
                }
                for d, m, x, n in zip(dates, t2m_mean, t2m_max, t2m_min)
            ],
        ),
    }

    for base in targets:
        for filename, (fields, rows) in datasets.items():
            path = base / filename
            if path.exists():
                continue
            _write_csv_rows(path, fields, rows)
            print(f"🧪 Seeded sample dataset → {path.relative_to(ROOT)}")


def ensure_region_setup(region: str, *, seed_samples: bool) -> None:
    profile_path = _ensure_region_profile(region)
    profile = _config().load_region_profile(region)
    region_meta = profile.get("region_meta", {})
    print(f"🗂  Using profile → {profile_path}")
    print(f"🏷  Region name: {region_meta.get('name', region)}")

    cfg = _config()
    cfg.ensure_region_workspace(region)
    cfg.get_region_data_root(region)
    cfg.get_region_cache_dir(region)
    cfg.get_region_current_dir(region)

    if seed_samples:
        _seed_sample_fetches(region)


def _ensure_offline_region_setup(region: str) -> dict[str, object]:
    """Create minimal region scaffolding without external dependencies."""

    profile_dir = ROOT / "regions" / "profiles"
    profile_dir.mkdir(parents=True, exist_ok=True)
    profile_path = profile_dir / f"insight.{region}.yml"

    default_metadata = {
        "region_name": region.replace("_", " ").title(),
        "crops": ["coffee"],
        "country": "Unknown",
    }

    if not profile_path.exists():
        content = [
            "region_meta:",
            f"  region_name: {default_metadata['region_name']}",
            f"  crops:",
            f"    - {default_metadata['crops'][0]}",
            "  country: Unknown",
        ]
        profile_path.write_text("\n".join(content) + "\n", encoding="utf-8")

    data_root = ROOT / "data" / region
    current_dir = data_root / "current"
    cache_dir = data_root / "caches"
    for path in (data_root, current_dir, cache_dir):
        path.mkdir(parents=True, exist_ok=True)

    workspace = ROOT / "regions" / "workspaces" / region
    (workspace / "models").mkdir(parents=True, exist_ok=True)
    (workspace / "logs").mkdir(exist_ok=True)

    _seed_sample_fetches(region)

    return default_metadata


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _write_csv_rows(path: Path, fieldnames: Sequence[str], rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _fallback_build_caches(region: str) -> list[dict[str, object]]:
    current_dir = ROOT / "data" / region / "current"

    sources = {
        "chirps_gee.csv": ("precip_mm_sum",),
        "soil_gee.csv": ("soil_surface_moisture", "soil_rootzone_moisture"),
        "ndvi_gee.csv": ("ndvi", "ndvi_anomaly"),
        "openmeteo.csv": ("t2m_mean", "t2m_max", "t2m_min"),
    }

    combined: dict[str, dict[str, float]] = {}
    for filename, columns in sources.items():
        for row in _read_csv_rows(current_dir / filename):
            date = row.get("date")
            if not date:
                continue
            bucket = combined.setdefault(date, {})
            for column in columns:
                value = row.get(column)
                if value is None or value == "":
                    continue
                try:
                    bucket[column] = float(value)
                except ValueError:
                    continue

    daily_rows = []
    for date in sorted(combined.keys()):
        entry = {"date": date}
        entry.update(combined[date])
        daily_rows.append(entry)

    daily_path = current_dir / "daily_merged.csv"
    fieldnames = ["date"] + sorted({key for row in daily_rows for key in row.keys() if key != "date"})
    _write_csv_rows(daily_path, fieldnames, daily_rows)

    anomalies_path = current_dir / "daily_anomalies.csv"
    _write_csv_rows(anomalies_path, fieldnames, daily_rows)

    grouped: dict[str, dict[str, list[float]]] = {}
    for row in daily_rows:
        month = row["date"][0:7]
        bucket = grouped.setdefault(month, {key: [] for key in fieldnames if key != "date"})
        for key, value in row.items():
            if key == "date":
                continue
            bucket.setdefault(key, []).append(float(value))

    monthly_rows: list[dict[str, object]] = []
    for month in sorted(grouped.keys()):
        entry: dict[str, object] = {"month": month}
        for key, values in grouped[month].items():
            if values:
                entry[key] = round(statistics.fmean(values), 4)
        monthly_rows.append(entry)

    monthly_path = current_dir / "monthly_merged.csv"
    monthly_fieldnames = ["month"] + sorted({key for row in monthly_rows for key in row.keys() if key != "month"})
    _write_csv_rows(monthly_path, monthly_fieldnames, monthly_rows)

    return monthly_rows


def _fallback_generate_insights(
    region: str,
    monthly_rows: list[dict[str, object]],
    metadata: dict[str, object],
) -> Path:
    output_dir = Path("outputs") / region
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "distilled_summary.csv"
    fieldnames = ["month"] + sorted({key for row in monthly_rows for key in row if key != "month"})
    _write_csv_rows(summary_path, fieldnames, monthly_rows)

    feed_path = output_dir / "insight_feed.csv"
    region_name = metadata.get("region_name") or region.replace("_", " ").title()
    crop_list = metadata.get("crops") or ["mixed cropping"]
    crop = crop_list[0] if isinstance(crop_list, list) and crop_list else "mixed cropping"

    feed_rows = []
    alerts: list[str] = []
    for row in monthly_rows:
        month = row.get("month")
        spi = float(row.get("precip_mm_sum", 0.0))
        ndvi = float(row.get("ndvi_anomaly", 0.0))
        soil = float(row.get("soil_surface_moisture", 0.0))
        temp = float(row.get("t2m_mean", 0.0))
        risk = max(0.0, min(1.0, 0.5 - ndvi))
        text = (
            f"{region_name} ({crop}) — {month}: precipitation {spi:.2f} mm, "
            f"NDVI anomaly {ndvi:.3f}, soil moisture {soil:.2f}, temperature {temp:.1f} °C."
        )
        feed_rows.append(
            {
                "month": month,
                "crop_type": crop,
                "region_name": region_name,
                "spi": round(spi, 3),
                "ndvi_anomaly": round(ndvi, 3),
                "soil_surface_moisture": round(soil, 3),
                "temp_mean": round(temp, 3),
                "rule_hits": "",
                "model_signal": round(risk, 3),
                "insight_text": text,
            }
        )
        if risk >= 0.6:
            alerts.append(f"[{region}] {month}: Elevated vegetation stress risk (score={risk:.2f})")

    _write_csv_rows(
        feed_path,
        [
            "month",
            "crop_type",
            "region_name",
            "spi",
            "ndvi_anomaly",
            "soil_surface_moisture",
            "temp_mean",
            "rule_hits",
            "model_signal",
            "insight_text",
        ],
        feed_rows,
    )

    if alerts:
        alerts_path = output_dir / "alerts.txt"
        alerts_path.write_text("\n".join(alerts) + "\n", encoding="utf-8")

    return summary_path


def _fallback_train_model(region: str, monthly_rows: list[dict[str, object]]) -> Path:
    scores = []
    for row in monthly_rows:
        ndvi = float(row.get("ndvi_anomaly", 0.0))
        soil = float(row.get("soil_surface_moisture", 0.0))
        precip = float(row.get("precip_mm_sum", 0.0))
        score = max(0.0, min(1.0, 0.5 - 0.4 * ndvi + 0.1 * (0.3 - soil) + 0.05 * (0.5 - precip / 20.0)))
        scores.append(score)

    avg_score = sum(scores) / len(scores) if scores else 0.0
    model_data = {
        "type": "fallback",
        "average_score": avg_score,
        "history": scores,
    }

    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"{region}_rf.pkl"
    model_path.write_text(json.dumps(model_data, indent=2), encoding="utf-8")

    metrics_path = Path("outputs") / region / "model_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_payload = {
        "region": region,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "metrics": {
            "r2": 0.0,
            "mae": 0.0,
            "rmse": 0.0,
            "n": len(scores),
        },
        "notes": "Fallback heuristic model used due to missing ML dependencies.",
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    return model_path


def _run_offline_fallback(region: str, missing: Sequence[str]) -> int:
    print(
        "⚠️  Running simplified offline pipeline due to missing dependencies: "
        + ", ".join(missing)
    )

    metadata = _ensure_offline_region_setup(region)
    monthly_rows = _fallback_build_caches(region)
    summary_path = _fallback_generate_insights(region, monthly_rows, metadata)
    model_path = _fallback_train_model(region, monthly_rows)

    print(f"✅ Offline distillation complete → {summary_path}")
    print(f"✅ Offline model artifacts saved → {model_path}")
    return 0


# ---------------------------------------------------------------------------
# CLI & main orchestration
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the regional insights pipeline.")
    parser.add_argument("--region", required=True, help="Region slug to operate on (e.g. hungary_farmland)")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run the full pipeline (setup, fetch, cache, context, insights, train, evaluate).",
    )
    parser.add_argument("--offline", dest="offline", action="store_true", help="Avoid installing new dependencies.")
    parser.add_argument("--online", dest="offline", action="store_false", help="Allow dependency installation.")
    parser.set_defaults(offline=True)
    parser.add_argument(
        "--fetch-mode",
        choices=["active", "cached"],
        default="cached",
        help="Fetcher mode forwarded to scripts/fetch_all.py.",
    )
    parser.add_argument("--skip-setup", action="store_true", help="Skip workspace/config setup.")
    parser.add_argument("--skip-fetch", action="store_true", help="Skip data fetch stage.")
    parser.add_argument("--skip-cache", action="store_true", help="Skip cache build stage.")
    parser.add_argument("--skip-context", action="store_true", help="Skip context layer build stage.")
    parser.add_argument("--skip-insights", action="store_true", help="Skip insight distillation stage.")
    parser.add_argument("--skip-train", action="store_true", help="Skip model training stage.")
    parser.add_argument("--skip-evaluate", action="store_true", help="Skip model evaluation stage.")
    parser.add_argument(
        "--seed-samples",
        action="store_true",
        help="Create synthetic fetch outputs for offline development runs.",
    )
    parser.add_argument(
        "--allow-fetch-failures",
        action="store_true",
        help="Continue even if fetch_all.py exits with an error (useful offline).",
    )
    parser.add_argument(
        "--allow-context-failures",
        action="store_true",
        help="Continue even if build_context_layers.py exits with an error.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.full:
        args.skip_setup = False
        args.skip_fetch = False
        args.skip_cache = False
        args.skip_context = False
        args.skip_insights = False
        args.skip_train = False
        args.skip_evaluate = False
        if not args.seed_samples:
            args.seed_samples = True
        args.offline = True
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    if args.offline:
        os.environ.setdefault("OFFLINE_MODE", "1")

    missing = bootstrap_dependencies(ESSENTIAL_PACKAGES)

    if missing and any(name in missing for name in ("pandas", "numpy", "scikit-learn")):
        return _run_offline_fallback(args.region, missing)

    if not args.skip_setup:
        ensure_region_setup(args.region, seed_samples=args.seed_samples)

    steps = [
        StepConfig(
            name="Fetch datasets",
            command=build_command("fetch_all.py", "--region", args.region, "--mode", args.fetch_mode),
            allow_failure=args.allow_fetch_failures or args.offline,
            skip=args.skip_fetch,
        ),
        StepConfig(
            name="Build region cache",
            command=build_command("build_region_cache.py", "--region", args.region),
            skip=args.skip_cache,
        ),
        StepConfig(
            name="Build context layers",
            command=build_command("build_context_layers.py", "--region", args.region),
            allow_failure=args.allow_context_failures or args.offline,
            skip=args.skip_context,
        ),
        StepConfig(
            name="Build region insights",
            command=build_command("build_region_insights.py", "--region", args.region),
            skip=args.skip_insights,
        ),
        StepConfig(
            name="Train regional model",
            command=(
                sys.executable,
                str(SCRIPTS_DIR / "train_region_model.py"),
                "--region",
                args.region,
                "--tier",
                "1",
                "--freq",
                "monthly",
                "--target",
                "ndvi_zscore",
            ),
            skip=args.skip_train,
        ),
        StepConfig(
            name="Evaluate model effectiveness",
            command=(
                sys.executable,
                str(SCRIPTS_DIR / "evaluate_effectiveness.py"),
                "--region",
                args.region,
                "--tier",
                "1",
                "--freq",
                "monthly",
                "--target",
                "ndvi_zscore",
            ),
            skip=args.skip_evaluate,
            allow_failure=False,
        ),
    ]

    for step in steps:
        run_step(step)

print("\n🎉 Pipeline completed successfully for", region)
print("📊 Outputs saved to:", f"outputs/{region}/")
print("🧠 Model artifacts saved to:", f"models/{region}_rf.pkl")

print("""
💡 Next Steps:
- Add or update local practice logs in `data/<region>/practice_logs/`
  to help the model learn from real field data.
- Re-run `train_region_model.py` to retrain on updated logs.
- Then re-run `run_pipeline.py` to see improved, context-aware insights.
""")
print("\n💬 Example insight from latest run:")
try:
    import pandas as pd
    df = pd.read_csv(f"outputs/{region}/distilled_summary.csv")
    example = df.iloc[-1]["insight_text"]
    print(f"🧠 {example}")
except Exception:
    print("🧠 (Insight text unavailable — check distilled_summary.csv)")

print("""
💡 Next Steps:
- Add recent practice logs in `data/<region>/practice_logs/` (e.g., planting, irrigation, composting).
- Re-run training with `python scripts/train_region_model.py --region <region>`
  to let the model learn from your field updates and refine future insights.
""")

return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
