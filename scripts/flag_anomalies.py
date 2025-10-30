import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import pandas as pd, numpy as np, yaml, json
from datetime import timedelta
from pathlib import Path
import operator

from _shared import resolve_region_config_path

OPS = {
    "<": operator.lt, "<=": operator.le, ">": operator.gt, ">=": operator.ge, "==": operator.eq, "!=": operator.ne
}

def load_config(path):
    config_path = Path(path)
    cfg = yaml.safe_load(config_path.read_text())
    if "extends" in cfg:
        extend_path = Path(cfg["extends"])
        if not extend_path.is_absolute():
            extend_path = config_path.parent / extend_path
        base = yaml.safe_load(extend_path.read_text())
        # shallow merge (good enough for our simple shapes)
        base.update({k:v for k,v in cfg.items() if k!="extends"})
        cfg = base
    return cfg

def eval_clause(row, clause):
    var, op, val = clause["var"], clause["op"], clause["value"]
    if var not in row or pd.isna(row[var]): return False
    fn = OPS[op]
    return bool(fn(row[var], val))

def eval_when(row, tree):
    # Supports {"all":[clauses]} and/or {"any":[clauses]} nesting one level
    if "all" in tree:
        if not all(eval_clause(row, c) for c in tree["all"]): return False
    if "any" in tree:
        if not any(eval_clause(row, c) for c in tree["any"]): return False
    return True

def main(region: str, config_path: str | None = None):
    rp = Path(f"data/{region}")
    daily_paths = [
        rp / "insights_daily.csv",
        rp / "insight_daily.csv",
        rp / "daily_merged.csv",
    ]
    for candidate in daily_paths:
        if candidate.exists():
            df = pd.read_csv(candidate, parse_dates=["date"]).sort_values("date")
            break
    else:
        raise FileNotFoundError(
            f"No daily insight/cache file found for {region}. Looked for: {', '.join(str(p) for p in daily_paths)}"
        )

    if config_path is None:
        config_path = resolve_region_config_path(region)
    cfg = load_config(str(config_path))

    rules = cfg.get("rules", [])
    flags = []
    for _, row in df.iterrows():
        for rule in rules:
            if eval_when(row, rule["when"]):
                flags.append({
                    "date": row["date"].date().isoformat(),
                    "rule_id": rule["id"],
                    "label": rule.get("label", rule["id"]),
                    "variables": {k: row[k] for k in row.index if k not in ("date",)}
                })

    # Write CSV + JSON + short MD digest
    out_dir = rp/"flags"
    out_dir.mkdir(parents=True, exist_ok=True)

    if flags:
        flags_df = pd.DataFrame(flags)
        flags_df.to_csv(out_dir/"flagged_anomalies.csv", index=False)
        Path(out_dir/"flagged_anomalies.json").write_text(json.dumps(flags, indent=2))

        # Markdown digest (last 30 days)
        last_date = pd.to_datetime(df["date"].max()).date()
        cutoff = pd.Timestamp(last_date - timedelta(days=30)).date()
        recent = [f for f in flags if pd.to_datetime(f["date"]).date() >= cutoff]

        lines = [f"# Anomaly Digest — {region}",
                 f"_Window: last 30 days through {last_date}_",
                 ""]
        if recent:
            for f in recent:
                lines.append(f"- **{f['date']}** — **{f['label']}** (`{f['rule_id']}`)")
        else:
            lines.append("No anomalies in the last 30 days.")
        (out_dir/"digest_recent.md").write_text("\n".join(lines))
        print(f"✅ Flags written → {out_dir}")
    else:
        print("✅ No anomalies flagged. Nothing to write.")
        (out_dir/"flagged_anomalies.csv").write_text("date,rule_id,label,variables\n")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--region", required=True)
    p.add_argument("--config", help="Optional path to a custom insight profile")
    a = p.parse_args()
    main(a.region, a.config)
