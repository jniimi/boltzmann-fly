"""Default locations. Large data and checkpoints live outside the repository."""
from __future__ import annotations

import os
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[2]

# Derived connectome data (masks, cached simulation datasets)
DERIVED_DIR = Path(os.environ.get("BOLTZMANN_FLY_DERIVED", "/Volumes/EXTERNAL/malecns/derived/boltzmann-fly"))
MASK_DIR = DERIVED_DIR / "masks"
DATA_DIR = DERIVED_DIR / "data"

# Trained weights / run artefacts
MODEL_DIR = Path(os.environ.get("BOLTZMANN_FLY_MODELS", "/Volumes/EXTERNAL/models/boltzmann-fly"))

# Small, versioned results inside the repo
RESULTS_DIR = REPO_DIR / "docs" / "results"
RUNS_DIR = RESULTS_DIR / "runs"
RESULTS_CSV = RESULTS_DIR / "results.csv"

NODES_PARQUET = DERIVED_DIR / "mb_nodes.parquet"
EDGES_PARQUET = DERIVED_DIR / "mb_edges.parquet"
