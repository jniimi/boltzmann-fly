#!/usr/bin/env python
"""Build the connectome masks (V1 / V1b / V2 x none / degree / er) for one hemisphere.

    uv run python scripts/build_masks.py --variant all --control all --seed 0
    uv run python scripts/build_masks.py --variant V1 --control none --seed 0 --type-select top-fanout

Writes <mask-dir>/<variant>_<control>_<side>_w<min-weight>_<type-select>_seed<seed>.{npz,json}
and prints a text table of unit / edge counts. No imagery is produced.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from boltzmann_fly.masks import CONTROLS, VARIANTS, build_maskset
from boltzmann_fly.paths import MASK_DIR


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", default="all", choices=("all",) + VARIANTS)
    ap.add_argument("--control", default="all", choices=("all",) + CONTROLS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--side", default="R")
    ap.add_argument("--min-weight", type=int, default=5)
    ap.add_argument("--type-select", default="random", choices=("random", "random-projecting", "top-fanout"),
                    help="random: any PN type (V1/V2) or PN (V1b); random-projecting: only types/PNs with a KC edge at "
                         "--min-weight; top-fanout: the 72 types with most KC targets")
    ap.add_argument("--mask-dir", type=Path, default=MASK_DIR)
    a = ap.parse_args()

    variants = VARIANTS if a.variant == "all" else (a.variant,)
    controls = CONTROLS if a.control == "all" else (a.control,)
    keys = ["n_visible", "n_visible_carrying_data", "n_features_connected", "n_pn_projecting", "n_pn_types_projecting",
            "n_kc", "n_kc_reached_by_data_units", "n_mbon",
            "edges_pn_kc", "edges_kc_mbon", "density_pn_kc", "density_kc_mbon", "n_couplings", "n_params_total",
            "overlap_with_real_pn_kc", "overlap_with_real_kc_mbon"]
    print("| variant | control | " + " | ".join(keys) + " |")
    print("|---|---|" + "---|" * len(keys))
    for v in variants:
        for c in controls:
            ms = build_maskset(v, c, a.seed, a.side, a.min_weight, a.type_select)
            p = ms.save(a.mask_dir)
            cells = [f"{ms.stats[k]:.4f}" if isinstance(ms.stats[k], float) else f"{ms.stats[k]:,}" for k in keys]
            print(f"| {v} | {c} | " + " | ".join(cells) + f" |   -> {p.name}")


if __name__ == "__main__":
    main()
