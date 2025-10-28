# Copilot Instructions for Regional Climate Agent

This guide helps AI agents understand and work effectively with this codebase.

## Project Architecture

### Core Components
- **Data Pipeline**
  - `scripts/`: Data fetchers and processors for climate data (CHIRPS, ERA5, OpenMeteo)
  - `engine/distill.py`: Core data distillation engine
  - `data/*/region.yml`: Region-specific configurations
  
- **Agent Logic**
  - `agent/`: Contains feature extraction, rule engine, and optional RAG retriever
  - `notebooks/`: Development and production notebooks for both training and inference

### Data Flow
1. Raw data → Cached CSVs (`data/*/chirps_cached.csv`, etc.)
2. Distillation → Regional features (Zarr format)
3. Agent processing → Insights and recommendation cards (JSON)

## Development Workflow

### Environment Setup
```bash
conda env create -f environment.yml  # Create environment
conda activate regional-agent  # Activate environment
```

### Key Commands
- Data refresh: `python scripts/build_region_cache.py`
- Training: `python scripts/build_training_window.py`
- Evaluation: `python engine/evaluate_greenai.py`

### Testing and Validation
Use `notebooks/regional_agent_dev.ipynb` for development testing. All notebooks are designed to be deterministic when run top-to-bottom.

## Project Conventions

### Data Organization
- Region-specific data in `data/<region_name>/`
- Each region requires `region.yml` config file
- Cache files use consistent naming: `{source}_cached.csv`

### Code Patterns
- Feature extraction in `agent/feature_extract.py` follows scikit-learn API
- All data processing must be CPU-only for reproducibility
- Green AI metrics logged via `codecarbon`

### Integration Points

#### Kaggle Integration
- Entry point: `notebooks/regional_agent_kaggle.ipynb`
- Automated dataset export via `.github/workflows/kaggle_export.yml`
- Dataset structure:
  - `/data/*/chirps_cached.csv`: Historical precipitation
  - `/data/*/era5_recent.csv`: Recent climate variables
  - `/data/*/openmeteo_cached.csv`: Weather forecasts
  - `/data/*/region.yml`: Region configuration

#### GitHub Actions Automation
- Cache refresh (weekly): `.github/workflows/cache.yml`
  - Updates CHIRPS, ERA5, and OpenMeteo data
  - Regenerates distilled features
- Training pipeline (monthly): `.github/workflows/training.yml`
  - Retrains models on expanded data window
  - Updates feature importance metrics
- Kaggle export: `.github/workflows/kaggle_export.yml`
  - Bundles `/data` directory for Kaggle Dataset
  - Maintains version history and metadata

#### Retrieval-Augmented Generation (RAG)
- Optional agronomic knowledge retrieval in `agent/rag_retriever.py`
- Document corpus seeding: `scripts/seed_docs_corpus.py`
- Integration with region-specific crop contexts from `region.yml`

#### External Data Sources
- CHIRPS: Global precipitation (daily, 0.05°)
- ERA5: Climate reanalysis (hourly, 0.25°)
- OpenMeteo: Weather forecasts (hourly)
- SoilGrids: Soil profiles (250m resolution)

## Documentation References
- Architecture details: `docs/caching_layers.md`
- Configuration guide: `docs/regional_config_guide.md`
- Green AI metrics: `docs/greenai_metrics.md`