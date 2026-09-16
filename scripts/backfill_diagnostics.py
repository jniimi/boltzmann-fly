#!/usr/bin/env python
"""Backfill belief / drive / bias-only diagnostics into run JSONs produced before these were recorded.

    uv run python scripts/backfill_diagnostics.py            # all runs lacking 'diagnostics'
Loads the pickled fine-tuned DBM of each run (CPU) and recomputes the test-split diagnostics.
"""
from __future__ import annotations

import glob
import json
import pickle
import sys
from pathlib import Path

import torch

import boltzmann_fly.masked_dbm  # noqa: F401  (unpickling)
from boltzmann_fly.data import ICONIP_DATA, FlyVisibleDataset, SimulationDataset, load_or_generate
from boltzmann_fly.energy import clamp_consistency
from boltzmann_fly.masks import MaskSet
from boltzmann_fly.paths import RUNS_DIR
from boltzmann_fly.pipeline import belief_stats, dense_drive_stats

DIAG_KEYS = ["belief_sd_median_l1", "belief_sd_median_l2", "belief_lowvar_frac_l1", "belief_lowvar_frac_l2",
             "drive_median_l1", "drive_median_l2"]


def main():
    dev = torch.device("cpu")
    _, _, te = load_or_generate(ICONIP_DATA, verbose=False)
    base_te = SimulationDataset(te)
    for f in sorted(glob.glob(str(RUNS_DIR / "*.json"))):
        r = json.load(open(f))
        row = r["row"]
        if "diagnostics" in r and row.get("energy_test_bias_corr") not in (None, ""):
            continue
        if row.get("smoke"):
            continue
        p = Path(r["drive_dir"]) / "models" / "dbm_finetuned.pkl"
        if not p.exists():
            print(f"skip {f}: no checkpoint"); continue
        dbm = pickle.load(open(p, "rb")).to(dev).eval()
        if row["variant"] == "V0":
            ds, emb, masked = base_te, None, hasattr(dbm, "mask_0")
        else:
            ms = MaskSet.load(row["variant"], row["control"], row["seed"], type_select=row.get("type_select", "random"))
            emb = ms.embedding_matrix(); ds = FlyVisibleDataset(base_te, emb); masked = True
        diag = belief_stats(dbm, ds, dev)
        diag.update(dbm.drive_stats() if masked else dense_drive_stats(dbm))
        et = clamp_consistency(dbm, ds, dev, embedding=emb, verbose=False)
        r["diagnostics"] = diag
        r["energy"]["test"] = et
        row.update({k: diag[k] for k in DIAG_KEYS})
        row["energy_test_bias_corr"] = et["bias_only_corr"]; row["energy_test_bias_only_mean"] = et["bias_only_mean"]
        row.setdefault("init", "default"); row.setdefault("lr_scale", 1.0); row.setdefault("tag", "")
        json.dump(r, open(f, "w"), indent=1)
        print(f"{row['run_id']}: sd_l1={diag['belief_sd_median_l1']:.4f} drive_l1={diag['drive_median_l1']:.3f} bias_corr={et['bias_only_corr']:.3f}")


if __name__ == "__main__":
    main()
