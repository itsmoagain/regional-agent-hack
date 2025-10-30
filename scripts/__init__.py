import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
"""Helper utilities for running project scripts as a package."""

# Re-export the dependency helper so scripts can import it without
# worrying about relative paths. The import is intentionally inside a
# try/except block so that package consumers do not need to have
# run_pipeline available during installation steps.
try:  # pragma: no cover - defensive import guard
    from .run_pipeline import require  # noqa: F401
except Exception:
    # Fallback for environments where run_pipeline has not been created yet
    # or where its dependencies are still being bootstrapped.  Scripts can
    # import :func:`require` lazily after run_pipeline exists.
    require = None  # type: ignore
