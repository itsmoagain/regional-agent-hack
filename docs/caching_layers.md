# Caching Layers

Regional caches now follow a snapshot-based layout:

```
data/<region>/
  caches/
    20241001T000000Z/
      chirps_gee.csv
      openmeteo.csv
      ndvi_gee.csv
      soil_gee.csv
      manifest.json
  current/
    … (copy of the active snapshot plus derived outputs)
```

Each layer declares TTL and provenance inside `config/insight.<region>.yml`. The snapshot manifest captures:

- `layer`, `cache_file`, and fetcher name
- `fetched_at` / `expires_at` timestamps (UTC)
- SHA256 hash, byte size, and row count for deterministic reuse
- Optional provenance metadata (source URL, fetch command timings)

`scripts/run_pipeline.py` enforces TTL policy:

- `--bootstrap` pulls every required layer and writes a fresh snapshot.
- `--refresh` reuses cached files unless the TTL has expired (unless `--allow-stale`).
- `--analyze` never hits the network and fails if stale required data exceed allowed thresholds.
