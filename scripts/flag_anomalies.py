import pandas as pd, numpy as np, yaml, json
from datetime import timedelta
from pathlib import Path
import operator

OPS = {
    "<": operator.lt, "<=": operator.le, ">": operator.gt, ">=": operator.ge, "==": operator.eq, "!=": operator.ne
}

def load_config(path):
    cfg = yaml.safe_load(Path(path).read_text())
    if "extends" in cfg:
        base = yaml.safe_load(Path(cfg["extends"]).read_text())
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

def main(region:str, config_path:str):
    rp = Path(f"data/{region}")
    df = pd.read_csv(rp/"insight_daily.csv", parse_dates=["date"]).sort_values("date")
    cfg = load_config(config_path)

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
    p.add_argument("--config", required=True)
    a = p.parse_args()
    main(a.region, a.config)
