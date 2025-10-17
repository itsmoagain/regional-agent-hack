# Situated Insight — Regional Climate Agent Pilot

**Goal:** Reproducible, CPU-only *Green AI* pipeline that distills global climate data into local insights for two regions:
- 🇭🇺 **Hungary Farmland** (temperate; wheat/maize)
- 🇯🇲 **Jamaica Coffee** (tropical; shaded arabica)

## Layers
- **Insight Layer (Kaggle/offline):** uses cached CSV/Zarr; *no network calls*.
- **Training Layer (GitHub/online):** lightweight monthly RF retrain using ERA5/NDVI windows.

## Reproducibility
- Version-locked deps (`requirements.txt`)
- YAML-driven region configs (`/data/*/region.yml`)
- Deterministic notebooks
- Green AI logging with `codecarbon`

## Run on Kaggle (offline)
1. Create a Kaggle Dataset from `/data` (or use `kaggle_export.yml` to bundle).
2. Open `notebooks/regional_agent_kaggle.ipynb` in Kaggle; attach the dataset.
3. Run top-to-bottom (CPU). Produces `outputs/insights/*_rec_cards.json`.

## Run on GitHub (online)
- **Cache refresh:** `.github/workflows/cache.yml` (weekly)
- **Training:** `.github/workflows/training.yml` (monthly, optional)
- **Kaggle export:** `.github/workflows/kaggle_export.yml` (bundles /data for upload)

## ASCII Flow
Global Datasets (CHIRPS, ERA5, MODIS)
    │
    ▼
Distillation Engine → regional_features.zarr
    │
    ▼
Agent Logic → rec_cards.json + insights.csv
    │
    └── Farmer Logs ↔ Enriched Data → Feedback Loop

See `docs/` for architecture, configuration, caching layers, and submission notes.
