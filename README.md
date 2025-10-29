# 🌍 Situated Insight — Regional Climate Agent Pilot

**Reproducible, open, and locally meaningful climate intelligence**

Situated Insight demonstrates how lightweight, reproducible AI systems can distill global climate data into locally relevant insights — enabling land stewards, cooperatives, and researchers to operate from shared, transparent context without relying on large compute resources or centralized infrastructure.

This pilot shows that climate intelligence can be both green and grounded: localized, explainable, and fully re-runnable on CPU-only environments.

---

## 🧠 Concept Overview

Each regional agent functions as a self-contained, evolving insight engine, built around a simple principle: global climate data should adapt to place.

The system converts high-volume, high-latency datasets into regional learning capsules — compact, transparent models that improve through local feedback and contextual enrichment.

Each capsule follows five coordinated steps:

1. **Data Distillation** – Convert terabyte-scale sources (CHIRPS, ERA5, MODIS) into lightweight regional caches.
2. **Feature Computation** – Derive interpretable climate indicators (SPI, GDD, NDVI, VPD) that directly reflect field conditions.
3. **Context Fusion** – Blend climate signals with regional or crop-specific metadata for grounded analysis.
4. **Regional Modelling** – Train interpretable Random Forest ensembles that capture local climate–crop dynamics and can interoperate with other regional models.
5. **Agent Reasoning & Insight Feed** – Translate model outputs into contextual summaries and threshold-based alerts — suitable for dashboards, cooperatives, or DAO networks.

Together, these local engines form a mosaic of regional intelligence — a decentralized climate-learning network that grows through recombination and reuse rather than centralized scaling.

---

## ♻️ Green AI Design

Green AI here refers to an approach that is contextually regenerative — computation, insight, and participation are designed to sustain one another.

Situated Insight treats climate intelligence as a living system. Each component, from data distillation to feedback dashboards, is built to minimize waste and maximize reciprocity across human, computational, and ecological layers.

1. **Local Distillation for Global Efficiency**
   → Each region runs a lightweight climate distillation, caching rainfall, temperature, and NDVI in compact, open formats (CSV/Zarr).
   → These regional climate slides replace repetitive downloads, forming reproducible baselines that can be extended, remixed, or shared across seasons and projects.

2. **Composable Regional Models**
   → Random Forests and other interpretable models are trained within each region — small enough for local hardware yet interoperable as a federated ensemble.
   → Together they form a distributed climate intelligence network, supporting cross-region learning without centralized infrastructure.

3. **Feedback-First Human–Machine Loops**
   → Insights are built for decision-making: “this rainfall anomaly aligns with below-normal NDVI recovery.”
   → Feedback events enrich the training corpus, linking observation, action, and learning in continuous cycles.

4. **Participatory and Regenerative Data Flows**
   → Contributors retain agency over their data. Logs can be enriched locally, validated by peers, and shared voluntarily into open mosaics.
   → The system values context and contribution over scale, fostering shared learning and equitable participation.

In essence: Green AI here means a system that learns efficiently, acts locally, and evolves through connection.

---

## 🧩 Pilot Regions

| Region | Context | Focus |
|--------|----------|--------|
| 🇭🇺 Hungary Farmland | Temperate, mixed cropping | Climate variability and soil-moisture response |
| 🇯🇲 Jamaica Coffee Belt | Tropical, high-elevation | Shade dynamics and rainfall anomalies |

---

## 🌐 Regional Insight Mesh

The long-term vision is a network of regional agents — each a climate node trained on localized data yet interoperable through shared metadata and features.

Each region distills global datasets into its own cache: CHIRPS rainfall, ERA5 reanalysis, MODIS NDVI, and local context.

Each cache trains its own lightweight model, producing interpretable regional logic.

These regional models can link or aggregate into a global insight mesh — a distributed knowledge fabric that strengthens through reuse and interconnection.

Capabilities include:

- Reduced energy footprint through one-time regional computation and long-term reuse.
- Participatory enrichment where cooperatives and researchers contribute local data or retraining triggers.
- Interoperability between regional models, enabling shared queries and comparative analysis.

The mesh expands organically, forming a planetary network of small, efficient climate AIs whose combined insight grows with every new region added.

---

## ⚙️ Architecture

```
(Global datasets)
      │
      ▼
[Regional Distillation Engine]
      │
      ▼
 [Local Cache: CHIRPS, ERA5, MODIS]
      │
      ▼
 [Derived Features: SPI, GDD, NDVI, VPD]
      │
      ▼
 [Crop & Soil Context]
      │
      ▼
 [Agent Reasoning]
      │
      ▼
 [Insight Cards / Emissions Log]
```

---

## 🧭 Setup & Reproducibility

```bash
git clone https://github.com/itsmoagain/regional-agent-hack.git
cd regional-agent-hack
pip install -r requirements.txt
python scripts/build_region_cache.py --region hungary_farmland --track
```

### 🚀 Run the end-to-end pipeline

You can run the full regional workflow (initialization → fetch → cache → insights → model training) with a single command:

```bash
python scripts/run_pipeline.py --region hungary_farmland
```

Add flags such as `--skip-fetch`, `--skip-train`, or `--report reports/hungary_pipeline.json` to customize what runs and to capture a machine-readable summary of each stage.

Recommended Python: 3.12.x  
Compatible with: Kaggle notebooks and GitHub Actions CPU runners
