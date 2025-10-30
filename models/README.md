# Model artifacts

This directory stores locally generated checkpoints, metrics, and feature reports
produced by the regional modelling pipeline. Binary files (for example
`*_model.pkl`) are intentionally **not** tracked in git. Recreate them by running
one of the training entry points, such as:

```bash
python scripts/train_region_model.py --region <slug> --tier <n> --freq monthly
```

To keep the repository lean, avoid committing large binaries or derived outputs.
If you need to persist a model for collaboration, upload it to an object store or
attach it to the relevant issue instead of versioning it here.
