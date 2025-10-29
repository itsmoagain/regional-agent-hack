# Region Workspaces

This folder separates reusable package code from the sample regions that ship with the template.

- `profiles/` holds `insight.<region>.yml` descriptors used by the setup wizard and the automation pipeline.
- `workspaces/` is where new areas of interest are scaffolded when you run the onboarding CLI. Each workspace
  mirrors the structure under `data/<region>/` and stores intermediate caches, plots, anomaly flags, and
  trained models.

The CLI and helper scripts load profiles directly from `regions/profiles/`, while the `regional_agent`
package mirrors outputs into `regions/workspaces/<region>/` so analysts have a clean, dedicated sandbox
after the automated pipeline finishes.
