#!/usr/bin/env python
"""Run one (variant, control, seed) through the full Purchase-World pipeline.

    uv run python scripts/run_experiment.py --variant V0 --seed 0 --device mps
    uv run python scripts/run_experiment.py --variant V1 --control degree --seed 0
    uv run python scripts/run_experiment.py --variant V1 --seed 0 --smoke      # 64 consumers, 2 epochs

Outputs: docs/results/runs/<run_id>.json, one row in docs/results/results.csv, and the trained
weights / log under /Volumes/EXTERNAL/models/boltzmann-fly/<run_id>/ (outside the repository).
"""
from __future__ import annotations

import argparse

from boltzmann_fly.masks import CONTROLS, VARIANTS
from boltzmann_fly.pipeline import run


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", required=True, choices=("V0",) + VARIANTS)
    ap.add_argument("--control", default="none", choices=CONTROLS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto", choices=("auto", "mps", "cuda", "cpu"))
    ap.add_argument("--epochs-pt", type=int, default=100, help="greedy pretraining epochs per layer (ICONIP: 100)")
    ap.add_argument("--epochs-ft", type=int, default=300, help="joint PCD fine-tuning epochs (ICONIP: 300, patience 20)")
    ap.add_argument("--adapter-epochs", type=int, default=100, help="adapter / baseline MLP epochs (ICONIP: 100, patience 30)")
    ap.add_argument("--dale", action="store_true", help="fix coupling signs by the presynaptic unit (|W| * sign)")
    ap.add_argument("--type-select", default="random", choices=("random", "random-projecting", "top-fanout"))
    ap.add_argument("--side", default="R")
    ap.add_argument("--min-weight", type=int, default=5)
    ap.add_argument("--smoke", action="store_true", help="64 consumers and 2 epochs per phase")
    ap.add_argument("--smoke-consumers", type=int, default=64)
    ap.add_argument("--force-masked", action="store_true", help="run V0 through MaskedDBM with all-ones masks")
    ap.add_argument("--rep", type=int, default=0, help="replicate index (distinct run_id for repeated identical runs)")
    ap.add_argument("--threads", type=int, default=None, help="torch CPU threads (fix it for bit-identical CPU replicates)")
    ap.add_argument("--no-deterministic", action="store_true", help="do not call torch.use_deterministic_algorithms")
    ap.add_argument("--init", default="default", choices=("default", "fanin"),
                    help="fly variants only: 'fanin' = W_ij ~ N(0, 1/fan_in_j) so the initial KC drive is O(1)")
    ap.add_argument("--lr-scale", type=float, default=1.0, help="fly variants only: multiply both Adam learning rates")
    ap.add_argument("--tag", default="", help="suffix for the run_id (diagnostic runs; excluded from the summary table)")
    ap.add_argument("--verbose", type=int, default=1)
    a = ap.parse_args()

    if a.smoke:
        a.epochs_pt = min(a.epochs_pt, 2)
        a.epochs_ft = min(a.epochs_ft, 2)
        a.adapter_epochs = min(a.adapter_epochs, 3)
    run(a.variant, a.control, a.seed, a.device, a.epochs_pt, a.epochs_ft, a.adapter_epochs, a.dale, a.type_select,
        a.side, a.min_weight, a.smoke, a.smoke_consumers, a.force_masked, a.verbose,
        rep=a.rep, threads=a.threads, deterministic=not a.no_deterministic, init=a.init, lr_scale=a.lr_scale, tag=a.tag)


if __name__ == "__main__":
    main()
