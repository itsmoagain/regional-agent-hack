# 🌍 Situated Insight — Regional Climate Agent Pilot
**Reproducible, open, and locally meaningful climate intelligence**

The *Situated Insight* project shows how lightweight, reproducible AI systems can distill global climate data into **locally relevant insights** — helping land stewards, cooperatives, and researchers work from the same context without requiring massive compute power or centralized servers.

Our pilot demonstrates that **climate intelligence can be both green and grounded**: localized, transparent, and powered by CPU-only pipelines that anyone can re-run.

---

## 🧠 Concept Overview
Each regional agent acts as a **self-contained, evolving climate insight engine**, built around the principle that global climate data should *adapt to place*.  
The system turns high-volume, high-latency global datasets into *regional learning capsules* — small, fast, transparent models that continuously improve through local enrichment.

Each capsule follows five core steps:

1. **Data Distillation** – Convert terabyte-scale sources (CHIRPS, ERA5, MODIS) into lightweight local datasets.  
2. **Feature Computation** – Extract interpretable signals (SPI, GDD, NDVI, VPD) that directly reflect field conditions.  
3. **Context Fusion** – Blend those signals with regional or crop-specific data for grounded analysis.  
4. **Random Forest Modelling** – Train region-tuned ensemble models that capture local climate–crop dynamics and can be recombined into a *global mesh of interpretable regional models*.  
5. **Agent Reasoning & Insight Feed** – Translate outputs into contextual summaries and threshold-based alerts — usable by cooperatives, dashboards, or DAO networks.

Together, these local engines form a **“mosaic” of regional intelligence** — a decentralized climate-learning network that grows by recombining local models, rather than scaling centralized compute.

---

## ♻️ Green AI Design
Green AI here doesn’t just mean *energy efficient*.  
It means **computationally regenerative** — maximizing insight per watt, per byte, per decision.  
The system’s design reduces both *climate modeling waste* and *decision latency* through three interlocking strategies:

1. **Local Distillation for Global Efficiency**  
   → Instead of repeatedly processing petabytes of reanalysis data, each region runs a one-time distillation — caching rainfall, temperature, and NDVI locally in compact, open formats (CSV/Zarr).  
   → This creates “climate slides” that can be reused, versioned, and extended across years or projects without rerunning full downloads.

2. **Composable Regional Models**  
   → Random Forests trained in each region are small enough to run on CPUs, yet collectively form a *distributed ensemble* — a green, federated alternative to energy-hungry large models.  
   → They can be merged, compared, or retrained incrementally to build a global climate-learning network without centralized compute or cloud lock-in.

3. **Human–Machine Efficiency Loop**  
   → Every model serves a *decision-support function* — not abstract prediction.  
   → Farmers, DAOs, or research networks get immediate insight: “this rainfall anomaly aligns with below-normal NDVI recovery,” creating feedback that improves adaptation strategies *and* future training data.

The result is a system that turns global climate data into *regional intelligence assets* — **small, composable, human-useful models that waste nothing**: not data, not compute, not insight.

---

## 🧩 Pilot Regions

| Region | Context | Focus |
|--------|----------|--------|
| 🇭🇺 **Hungary Farmland** | Temperate, mixed cropping | Climate variability and soil moisture response |
| 🇯🇲 **Jamaica Coffee Belt** | Tropical, high-elevation | Shade dynamics and rainfall anomalies |

---

## 🌐 Regional Insight Mesh
The long-term vision is a **network of regional agents**, each acting as a “climate node” trained on localized data, yet interoperable through shared metadata and features.

- Each region distills global datasets into its own **climate cache** — CHIRPS rainfall, ERA5 reanalysis, NDVI composites, and local context.  
- Each cache trains its own lightweight Random Forest model, producing interpretable regional logic.  
- These regional models can then be **linked or aggregated** into a *global insight mesh* — a distributed knowledge fabric that learns through connection and reuse.  

This architecture enables:
- **Energy reduction** through one-time regional computation and long-term reuse.  
- **Participatory enrichment** by allowing cooperatives and researchers to contribute local data or retraining triggers.  
- **Interoperability** between models, where shared climate and practice features can be queried like microservices.  

The mesh expands through regional connection and reuse, forming a **planetary network of small, efficient climate AIs** whose combined insight grows with every new region added.

---

## ⚙️ Architecture
(Global datasets) --> [Regional Distillation Engine] --> [Local Cache]
      |                           |
      v                           v
 [CHIRPS, ERA5, MODIS]       [SPI, GDD, NDVI, VPD Features]
      |                           |
      v                           v
 [Crop & Soil Data] --> [Agent Reasoning] --> [Insight Cards / Emissions Log]

---

## 🧭 Setup & Reproducibility
git clone https://github.com/itsmoagain/regional-agent-hack.git  
cd regional-agent-hack  
pip install -r requirements.txt  
python scripts/build_region_cache.py --region hungary_farmland --track  

> Recommended Python version: **3.12.x**  
> Compatible with Kaggle and GitHub Actions runners.

---

## ✳️ Credits
**Developed by Morgan Urich**  
Hack4Earth Green AI Olympiad 2025 – Budapest  
*Reproducible, ethical, and locally grounded climate computation.*
