# 🧭 Design Notes

## Purpose
Situated Insight is built to show that climate intelligence can be light, local, and reproducible — a regenerative data network instead of a single model. Every design decision supports accessibility, transparency, and reusability.

---

## Core Design Principles

**1. Regional Autonomy**  
Each agent operates as its own local climate node, producing region-specific baselines and models without cloud dependency.  
→ Enables offline runs, regional governance, and transparent versioning.

**2. Computational Regeneration**  
Processing happens once per region, then gets reused and extended — conserving both compute cycles and human effort.  
→ CHIRPS, ERA5, and MODIS data are cached as open CSV/Zarr files that persist as shared baselines.

**3. Interpretable Modeling**  
Random Forests were chosen over deep models for clarity, reproducibility, and low energy demand.  
→ They expose decision logic directly to stewards and researchers.

**4. Feedback-First Insight Flow**  
Outputs are translated into contextual signals (e.g., “below-normal NDVI recovery following low rainfall”).  
→ This loop turns analysis into adaptive feedback for cooperatives or DAO dashboards.

**5. Participatory Enrichment**  
Local actors can append management logs, crop calendars, or soil observations that enrich the training corpus.  
→ Each contribution improves the model’s relevance while preserving contributor ownership.

---

## Data Flow Summary

1. **Global sources fetched** once → clipped by bounding box → stored in `/data/<region>/`.
2. **Features** (SPI, GDD, NDVI, VPD) computed via scripts in `/scripts/feature_builders/`.
3. **Regional cache** merged and serialized in open formats for reuse.
4. **Model training** runs locally, saving outputs to `/models/<region>_rf.pkl`.
5. **Insight generation** produces readable outputs (plots, summaries, alerts).
6. **Feedback hooks** allow cooperatives to retrain or enrich models with new local data.

---

## Why This Matters

Traditional climate ML pipelines over-compute — repeating the same data preparation for every study. Situated Insight demonstrates a green alternative: **distill once, learn many times.** It shows how open, composable regional models can form a global insight fabric that grows through connection, not consumption.

---

**Maintainer:** Morgan Urich  
**Hack4Earth Green AI Olympiad 2025 – Budapest**
