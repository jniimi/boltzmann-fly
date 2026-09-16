"""Purchase-World simulation data: deterministic generation, on-disk cache, fly-visible wrapper.

`SimulationDataset` is the binary-visible dataset of the original Purchase-World pipeline;
`FlyVisibleDataset` re-encodes the 72 Purchase-World features into the PN visible layer via a
fixed 0/1 embedding matrix.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .paths import DATA_DIR
from .vendor.generate_simulation_v2 import (
    SimulationConfig, generate_simulation_data, create_train_val_test_split, get_visible_cols, get_action_cols,
)

# ICONIP 2026 configuration (recovered from the original run's logged metadata)
ICONIP_DATA = dict(n_consumers=1024, n_days=365, n_stores=10, seed=42, ws=4, val_periods=30, test_periods=30)


class SimulationDataset(Dataset):
    """Binary-visible dataset for the Purchase-World panel (one row = consumer x day)."""

    def __init__(self, df: pd.DataFrame, feature_columns: List[str] = None):
        self.df = df.reset_index(drop=True)
        if feature_columns is None:
            self.feature_columns = get_visible_cols(df)
        else:
            self.feature_columns = feature_columns
        self.data = torch.tensor(self.df[self.feature_columns].values, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {'gbm_vector': self.data[idx]}

    @property
    def n_visible(self):
        return self.data.shape[1]

    def get_targets(self) -> Dict[str, torch.Tensor]:
        return {
            'visit': torch.tensor(self.df['visit'].values, dtype=torch.float32),
            'purchase': torch.tensor(self.df['purchase'].values, dtype=torch.float32),
        }

    def get_actions(self) -> torch.Tensor:
        action_cols = get_action_cols(self.df)
        return torch.tensor(self.df[action_cols].values, dtype=torch.float32)

    @property
    def action_columns(self) -> List[str]:
        return get_action_cols(self.df)

    @property
    def n_actions(self) -> int:
        return len(self.action_columns)

    def get_metadata(self) -> pd.DataFrame:
        cols = ['consumer_id', 'day', 'date', 'true_alpha', 'true_gamma', 'true_beta']
        available = [c for c in cols if c in self.df.columns]
        return self.df[available].copy()

    def get_feature_index_map(self) -> Dict[str, int]:
        return {col: i for i, col in enumerate(self.feature_columns)}


class FlyVisibleDataset(SimulationDataset):
    """Same panel, but `gbm_vector` is the PN-layer encoding v_fly = v_pw @ E (E is 0/1)."""

    def __init__(self, base: SimulationDataset, embedding: np.ndarray):
        self.df = base.df
        self.feature_columns = base.feature_columns
        self.pw_data = base.data
        self.embedding = torch.as_tensor(embedding, dtype=torch.float32)
        assert self.embedding.shape[0] == self.pw_data.shape[1]
        self.data = self.pw_data @ self.embedding

    def embed(self, v_pw: torch.Tensor) -> torch.Tensor:
        return v_pw @ self.embedding.to(v_pw.device)


def dataset_tag(cfg: dict = ICONIP_DATA) -> str:
    return f"pw_n{cfg['n_consumers']}_d{cfg['n_days']}_s{cfg['n_stores']}_ws{cfg['ws']}_seed{cfg['seed']}_v{cfg['val_periods']}_t{cfg['test_periods']}"


def load_or_generate(cfg: dict = ICONIP_DATA, data_dir: Path = DATA_DIR, verbose: bool = True
                     ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate the ICONIP panel deterministically (numpy legacy RNG, seed 42) and cache it as parquet."""
    out = data_dir / dataset_tag(cfg)
    files = {s: out / f"{s}.parquet" for s in ("train", "val", "test")}
    if all(f.exists() for f in files.values()):
        if verbose:
            print(f"  Using cached dataset {out}")
        return tuple(pd.read_parquet(files[s]) for s in ("train", "val", "test"))
    out.mkdir(parents=True, exist_ok=True)
    sim = SimulationConfig(n_consumers=cfg["n_consumers"], n_days=cfg["n_days"], n_stores=cfg["n_stores"],
                           seed=cfg["seed"], ws=cfg["ws"])
    df = generate_simulation_data(config=sim, output_dir=None, verbose=verbose)
    tr, va, te = create_train_val_test_split(df=df, test_days=cfg["test_periods"], val_days=cfg["val_periods"],
                                             output_dir=None)
    for s, d in zip(("train", "val", "test"), (tr, va, te)):
        d.to_parquet(files[s], index=False)
    (out / "config.json").write_text(json.dumps({**sim.to_dict(), "val_periods": cfg["val_periods"],
                                                 "test_periods": cfg["test_periods"]}, indent=2))
    return tr, va, te
