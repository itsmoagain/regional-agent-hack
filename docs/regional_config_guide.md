# Regional Config Guide

Each region now ships with an `insight.<region>.yml` file under `regions/profiles/` that **extends** `insight.defaults.yml`.

Minimum keys to override:

- `extends`: always `insight.defaults.yml` unless you build a custom base.
- `region_meta`
  - `name`
  - `country`
  - `bbox` → `[lon_min, lat_min, lon_max, lat_max]`
  - `crops` → values taken from `config/crop_library.yml`
- `baseline.start_year` / `baseline.end_year`
- `rules` → optional rule-based insight triggers (see examples in `regions/profiles/insight.hungary_transdanubia.yml`).

All other keys inherit from the defaults file. You can override window sizes or variable mappings when the regional context demands it (e.g., longer SPI windows for perennial crops).
