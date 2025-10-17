# 🌍 Situated Insight — Regional Climate Agent Pilot
**Reproducible, open, and locally meaningful climate intelligence**

The *Situated Insight* project shows how lightweight, reproducible AI systems can distill global climate data into **locally relevant insights** — helping land stewards, cooperatives, and researchers work from the same context without requiring massive compute power or centralized servers.

Our pilot demonstrates that **climate intelligence can be both green and grounded**: localized, transparent, and powered by CPU-only pipelines that anyone can re-run.

---

## 💡 Why This Matters
Climate data is abundant — but actionable context is not. Producers, policymakers, and researchers often work from disjointed sources that aren’t tuned to local conditions or farmer realities.

*Situated Insight* creates **region-specific “climate distillation engines”** that:
- Compress global data (like CHIRPS and ERA5) into **lightweight, reproducible local caches**.
- Fuse those caches with **practice and crop context** for real-time decision support.
- Operate **entirely on open data and CPU-only compute**, supporting a *Green AI* ethos of efficiency and equity.

This bridges two gaps:
1. **Technical** — enabling low-resource regions to access climate analytics.  
2. **Ethical** — prioritizing transparency, reproducibility, and inclusion in AI-enabled adaptation.

---

## 🧠 Concept Overview
Each regional agent acts as a **self-contained insight engine**, built around:
1. **Data Distillation** – Aggregate and downscale rainfall, temperature, and reanalysis data.  
2. **Feature Computation** – Compute drought, heat, and growth indicators (SPI, GDD, VPD, NDVI trends).  
3. **Context Fusion** – Join these indicators with soil, crop, or practice logs.  
4. **Rule-Based Reasoning** – Derive interpretable, traceable recommendations.  
5. **Green AI Metrics** – Log runtime, data volume, and CO₂ footprint per run.  

The result is a portable, inspectable regional model — a local climate “capsule” that can be shared, compared, and extended.

---

## 🧩 Pilot Regions

| Region | Context | Focus |
|--------|----------|--------|
| 🇭🇺 **Hungary Farmland** | Temperate, mixed cropping | Climate variability and soil moisture response |
| 🇯🇲 **Jamaica Coffee Belt** | Tropical, high-elevation | Shade dynamics and rainfall anomalies |

Each region’s cache is versioned independently, showing how **one framework can serve many ecosystems** — from arid grains to cloud-forest coffee.

---

## ⚙️ Architecture
(Global datasets) --> [Regional Distillation Engine] --> [Local Cache]
      |                           |
      v                           v
 [CHIRPS, ERA5, MODIS]       [SPI, GDD, NDVI, VPD Features]
      |                           |
      v                           v
 [Crop & Soil Data] --> [Agent Reasoning] --> [Insight Cards / Emissions Log]

All steps run offline or in GitHub Actions, ensuring reproducibility and low energy cost.

---

## ♻️ Green AI Design
- **CPU-only computation** — no GPUs, no external APIs during inference.  
- **Energy and CO₂ logging** via [CodeCarbon](https://mlco2.github.io/codecarbon/).  
- **Version-locked pipelines** for full transparency and reproducibility.  
- **Dual-region reproducibility** — Hungary and Jamaica run identical workflows, proving generality.  

---

## 🧰 Technical Stack
| Layer | Function | Key Tools |
|--------|-----------|-----------|
| Data Distillation | Aggregate & clean | `xarray`, `rioxarray`, `pandas` |
| Feature Extraction | SPI, GDD, NDVI, anomalies | `numpy`, `climate-indices` |
| Rule Engine | Human-readable logic | `yaml`, `pandas` |
| Emissions Tracking | CO₂eq + runtime logs | `codecarbon` |
| Visualization | Insight charts | `plotly`, `matplotlib` |

---

## 🧭 Setup & Reproducibility
git clone https://github.com/itsmoagain/regional-agent-hack.git  
cd regional-agent-hack  
pip install -r requirements.txt  
python scripts/build_region_cache.py --region hungary_farmland --track  

> Recommended Python version: **3.12.x**  
> Compatible with Kaggle and GitHub Actions runners.

---

## 🕒 Automated Workflows
| Workflow | Function | Trigger |
|-----------|-----------|----------|
| **cache.yml** | Refresh regional CHIRPS / ERA5 / Open-Meteo caches | Weekly (auto) |
| **kaggle_export.yml** | Package data snapshots for leaderboard | Manual |
| **training.yml** | Retrain Random Forests on enriched data | Monthly (optional) |

---

## 📦 Outputs
Each region produces:
- `chirps_cached.csv` — Daily rainfall  
- `openmeteo_cached.csv` — Temperature and RH  
- `era5_recent.csv` — Short-term context  
- `cache_manifest.json` — Runtime, size, emissions  
- (Optional) `emissions.csv` — Energy log for Green AI report  

---

## 🪴 Broader Vision
*Situated Insight* is a step toward **participatory, distributed climate intelligence** — where farmers and cooperatives co-own their data and regional models, rather than consuming top-down analytics.

This prototype shows that **insight can be small, local, and open — without sacrificing rigor or transparency.**

---

## ✳️ Credits
**Developed by Morgan Urich**  
Hack4Earth Green AI Olympiad 2025 – Budapest  
*Reproducible, ethical, and locally grounded climate computation.*
