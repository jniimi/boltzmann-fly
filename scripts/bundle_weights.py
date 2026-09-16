#!/usr/bin/env python
"""Assemble the weights release bundle (checkpoints, logs, run JSONs, masks, manifest, checksums, tarball).

    uv run python scripts/bundle_weights.py [--version v0.1] [--out /Volumes/EXTERNAL/models/boltzmann-fly/release]

For every run JSON in docs/results/runs/ (tagged diagnostic runs excluded) the bundle contains
<run_id>/dbm_finetuned.pkl, <run_id>/adapter_visit_*.pkl, <run_id>/adapter_purchase_*.pkl,
<run_id>/output.log, <run_id>/<run_id>.json and, for fly variants, masks/<stem>.npz + .json.
Top level: MANIFEST.md (files, sizes, sha256, state-dict sha256), SHA256SUMS (sha256sum format),
and <bundle>.tar.gz next to the directory. Nothing is drawn or embedded; text only.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import pickle
import shutil
import tarfile
from pathlib import Path

import boltzmann_fly.masked_dbm  # noqa: F401  (unpickling MaskedDBM)
from boltzmann_fly.masks import mask_stem
from boltzmann_fly.paths import MASK_DIR, MODEL_DIR, RUNS_DIR


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_state_dict(p: Path) -> str:
    obj = pickle.load(open(p, "rb"))
    h = hashlib.sha256()
    for k, t in obj.state_dict().items():
        h.update(t.cpu().numpy().tobytes())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", default="v0.1")
    ap.add_argument("--out", type=Path, default=MODEL_DIR / "release")
    ap.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    ap.add_argument("--mask-dir", type=Path, default=MASK_DIR)
    a = ap.parse_args()

    name = f"boltzmann-fly-weights-{a.version}"
    root = a.out / name
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    (root / "masks").mkdir()

    manifest = ["# boltzmann-fly weights " + a.version, "",
                "Pickled `torch.nn.Module` objects written by the vendored Purchase World training code. Load with",
                "`import boltzmann_fly.masked_dbm, pickle; dbm = pickle.load(open('.../dbm_finetuned.pkl','rb'))` after",
                "`uv sync` in the boltzmann-fly repository (no purchase-world checkout needed). Masks are 0/1 presence",
                "matrices derived from MaleCNS v1.0 (CC-BY 4.0) plus body ids; no connectome edge tables are redistributed.",
                "The state-dict sha256 is over the concatenated raw tensor bytes of `state_dict()` (device-independent).", "",
                "| run_id | file | size (MB) | sha256 (file) | sha256 (state dict) |", "|---|---|---|---|---|"]
    sums = []
    copied = []
    masks_done = set()
    for jf in sorted(glob.glob(str(a.runs_dir / "*.json"))):
        r = json.load(open(jf))
        row = r["row"]
        if row.get("tag") or row.get("smoke"):
            continue
        run_id = row["run_id"]
        src = Path(r["drive_dir"])
        dst = root / run_id
        dst.mkdir()
        files = [src / "models" / "dbm_finetuned.pkl"]
        files += sorted((src / "models").glob("adapter_visit_*.pkl")) + sorted((src / "models").glob("adapter_purchase_*.pkl"))
        files += [src / "output.log"]
        for f in files:
            if not f.exists():
                print(f"  warning: {f} missing"); continue
            shutil.copy2(f, dst / f.name)
            copied.append(dst / f.name)
        shutil.copy2(jf, dst / f"{run_id}.json")
        copied.append(dst / f"{run_id}.json")
        if row["variant"] != "V0":
            stem = mask_stem(row["variant"], row["control"], row["seed"], "R", int(row.get("min_weight") or 5),
                             row.get("type_select", "random"))
            if stem not in masks_done:
                for ext in (".npz", ".json"):
                    shutil.copy2(a.mask_dir / f"{stem}{ext}", root / "masks" / f"{stem}{ext}")
                    copied.append(root / "masks" / f"{stem}{ext}")
                masks_done.add(stem)
            manifest.append(f"| {run_id} | masks/{stem}.npz, masks/{stem}.json | - | see SHA256SUMS | - |")
        for f in sorted(dst.iterdir()):
            sd = sha256_state_dict(f) if f.suffix == ".pkl" else "-"
            manifest.append(f"| {run_id} | {run_id}/{f.name} | {f.stat().st_size / 1e6:.2f} | {sha256_file(f)} | {sd} |")
        print(f"  {run_id}: {len(list(dst.iterdir()))} files")

    for f in sorted(copied):
        sums.append(f"{sha256_file(f)}  {f.relative_to(root).as_posix()}")
    (root / "MANIFEST.md").write_text("\n".join(manifest) + "\n")
    (root / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    tar = a.out / f"{name}.tar.gz"
    with tarfile.open(tar, "w:gz") as t:
        t.add(root, arcname=name)
    n_files = sum(1 for _ in root.rglob("*") if _.is_file())
    size = sum(f.stat().st_size for f in root.rglob("*") if f.is_file()) / 1e6
    print(f"bundle: {root}  files={n_files}  size={size:.1f} MB")
    print(f"tarball: {tar}  size={tar.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
