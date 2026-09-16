"""End-to-end run: data -> (masked) DBM -> three tasks -> JSON + CSV row.

The DBM / adapter / baseline hyperparameters are the ones of the ICONIP 2026 experiment (recovered
from the original run's logged metadata): 1024 consumers x 365 days, 10 stores, lag window 4, data
seed 42, val/test 30 + 30 days; DBM 64-32-16, pretraining 100 epochs/layer (Adam 1e-4, wd 1e-3, PCD
k=1, batch 128), fine-tuning up to 300 epochs (Adam 1e-5, wd 1e-4, k=5, 10 mean-field iterations,
patience 20 on val recon BCE); adapters / baseline MLP 64-32-16 (Adam 5e-4, dropout 0.1, batch 256,
100 epochs, patience 30).
"""
from __future__ import annotations

import csv
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .data import ICONIP_DATA, FlyVisibleDataset, SimulationDataset, load_or_generate
from .energy import clamp_consistency
from .masked_dbm import masked_classes
from .masks import MaskSet, build_maskset
from .paths import MASK_DIR, MODEL_DIR, RESULTS_CSV, RUNS_DIR
from .vendor.adapter import run_adapter_pipeline, run_baseline_mlp
from .vendor.train import evaluate_dbm, joint_finetuning, pretrain_bm

ICONIP_HIDDEN = [64, 32, 16]
ADAPTER_HIDDEN = [64, 32, 16]

CSV_COLUMNS = [
    "variant", "control", "seed", "dale", "type_select", "smoke", "run_id", "timestamp", "device", "rep", "threads", "deterministic",
    "init", "lr_scale", "tag", "min_weight", "n_features_connected",
    "belief_sd_median_l1", "belief_sd_median_l2", "belief_lowvar_frac_l1", "belief_lowvar_frac_l2",
    "drive_median_l1", "drive_median_l2", "energy_test_bias_corr", "energy_test_bias_only_mean",
    "n_visible", "n_h1", "n_h2", "n_h3", "belief_dim", "n_couplings", "n_params",
    "energy_test_delta_mean", "energy_test_frac_pos", "energy_test_auc", "energy_test_paired_t",
    "energy_test_dfe_high_beta", "energy_test_dfe_low_beta", "energy_test_welch_t",
    "baseline_visit_auc", "baseline_purchase_auc",
    "adapter_visit_full_auc", "adapter_visit_top_auc", "adapter_purchase_full_auc", "adapter_purchase_top_auc",
    "best_visit_belief", "best_purchase_belief",
    "cate_push_visit_gamma", "cate_push_visit_alpha", "cate_push_visit_beta",
    "cate_sale1_purchase_alpha", "cate_sale1_purchase_beta", "cate_sale1_purchase_gamma",
    "dbm_test_recon_bce", "dbm_test_free_energy", "dbm_val_recon_bce_best",
    "epochs_pt", "epochs_ft", "adapter_epochs", "t_pretrain_s", "t_finetune_s", "t_baseline_s", "t_adapters_s",
    "t_energy_s", "t_total_s",
]


class _Tee:
    def __init__(self, file_handle, stream):
        self._file, self._stream = file_handle, stream

    def write(self, msg):
        self._stream.write(msg); self._file.write(msg); self._file.flush()

    def flush(self):
        self._stream.flush(); self._file.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


def pick_device(name: str = "auto") -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def training_config(drive_dir: Path, layer_sizes, device, epochs_pt: int, epochs_ft: int, batch_size: int = 128,
                    lr_scale: float = 1.0) -> Dict[str, Any]:
    """Training configuration dict expected by the vendored training loops, with the ICONIP values.

    lr_scale multiplies both Adam learning rates (fly variants only). Weight decay is left as is:
    torch Adam adds wd*w to the gradient before normalisation, so the shrinkage per step already
    scales with lr.
    """
    return {
        "device": device,
        "drive_dir": drive_dir,
        "dataset": {"n_visible": layer_sizes[0]},
        "bm": {
            "layer_sizes": list(layer_sizes),
            "batchsize": batch_size,
            "nepochs_greedy_pretraining": epochs_pt,
            "pretraining_weight_decay": 1e-3,
            "pretraining_lr": 1e-4 * lr_scale,
            "pretraining_ksteps": 1,
            "save_pretrain_id": "dbm_pretrain",
            "weight_scaling": None,
            "nepochs_joint_finetuning": epochs_ft,
            "finetuning_lr": 1e-5 * lr_scale,
            "finetuning_decay": 1e-4,
            "finetuning_ksteps": 5,
            "finetuning_niter": 10,
            "finetuning_grad_clip": None,
            "finetuning_patience": 20,
            "save_finetuning_id": "dbm_finetuned",
            "use_gbrbm": False,
            "sigma_init": 1.0,
            "learn_sigma": False,
            "compensate_biases": True,
        },
        "gpt": {},
        "adapter": {},
        "verbose": {"send_message": False},
    }


def run_key(variant, control, seed, dale=False, type_select="random", smoke=False, rep=0, init="default",
            lr_scale=1.0, tag="", min_weight=5) -> str:
    k = f"{variant}_{control}_seed{seed}"
    if variant != "V0" and min_weight != 5:
        k += f"_w{min_weight}"
    if init != "default":
        k += f"_init-{init}"
    if lr_scale != 1.0:
        k += f"_lr{lr_scale:g}"
    if rep:
        k += f"_rep{rep}"
    if tag:
        k += f"_{tag}"
    if dale:
        k += "_dale"
    if variant != "V0" and type_select != "random":
        k += f"_{type_select}"
    if smoke:
        k += "_smoke"
    return k


def _subset_consumers(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df[df["consumer_id"] < n].reset_index(drop=True)


def run(variant: str, control: str = "none", seed: int = 0, device: str = "auto", epochs_pt: int = 100,
        epochs_ft: int = 300, adapter_epochs: int = 100, dale: bool = False, type_select: str = "random",
        side: str = "R", min_weight: int = 5, smoke: bool = False, smoke_consumers: int = 64,
        force_masked: bool = False, verbose: int = 1, model_dir: Path = MODEL_DIR, mask_dir: Path = MASK_DIR,
        runs_dir: Path = RUNS_DIR, results_csv: Path = RESULTS_CSV, rep: int = 0, threads: Optional[int] = None,
        deterministic: bool = True, init: str = "default", lr_scale: float = 1.0, tag: str = "") -> Dict[str, Any]:
    t_start = time.time()
    dev = pick_device(device)
    if threads:
        torch.set_num_threads(int(threads))
    if deterministic:
        # exact on CPU; on MPS/CUDA ops without a deterministic kernel only warn
        torch.use_deterministic_algorithms(True, warn_only=True)
    if variant == "V0":
        assert init == "default" and lr_scale == 1.0, "V0 always uses the ICONIP initialisation and learning rates"
    key = run_key(variant, control, seed, dale, type_select, smoke, rep, init, lr_scale, tag, min_weight)
    drive_dir = model_dir / key
    (drive_dir / "models").mkdir(parents=True, exist_ok=True)
    log_f = open(drive_dir / "output.log", "w", encoding="utf-8")
    orig_out, orig_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = _Tee(log_f, orig_out), _Tee(log_f, orig_err)
    try:
        return _run(variant, control, seed, dev, epochs_pt, epochs_ft, adapter_epochs, dale, type_select, side,
                    min_weight, smoke, smoke_consumers, force_masked, verbose, drive_dir, mask_dir, runs_dir,
                    results_csv, key, t_start, rep, deterministic, init, lr_scale, tag)
    finally:
        sys.stdout, sys.stderr = orig_out, orig_err
        log_f.close()


def _run(variant, control, seed, dev, epochs_pt, epochs_ft, adapter_epochs, dale, type_select, side, min_weight,
         smoke, smoke_consumers, force_masked, verbose, drive_dir, mask_dir, runs_dir, results_csv, key, t_start, rep=0,
         deterministic=True, init="default", lr_scale=1.0, tag=""):
    print(f"=== boltzmann-fly run {key} on {dev} ({datetime.now().isoformat(timespec='seconds')}) ===")
    print(f"  torch {torch.__version__}, threads={torch.get_num_threads()}, deterministic={deterministic}")
    timings: Dict[str, float] = {}

    # ---------------- data (deterministic; identical to the ICONIP panel)
    df_tr, df_va, df_te = load_or_generate(ICONIP_DATA, verbose=verbose > 0)
    if smoke:
        df_tr, df_va, df_te = (_subset_consumers(d, smoke_consumers) for d in (df_tr, df_va, df_te))
    base_tr, base_va, base_te = SimulationDataset(df_tr), SimulationDataset(df_va), SimulationDataset(df_te)
    print(f"  train={len(base_tr):,} val={len(base_va):,} test={len(base_te):,} features={base_tr.n_visible}")

    # ---------------- masks / visible encoding
    set_seed(seed)
    maskset: Optional[MaskSet] = None
    if variant == "V0":
        layer_sizes = [base_tr.n_visible] + ICONIP_HIDDEN
        dbm_tr, dbm_va, dbm_te = base_tr, base_va, base_te
        masks, signs, embedding = None, None, None
        assert control == "none", "controls apply to fly variants only"
    else:
        try:
            maskset = MaskSet.load(variant, control, seed, side, min_weight, type_select, mask_dir)
        except FileNotFoundError:
            print("  mask file not found; building it now")
            maskset = build_maskset(variant, control, seed, side, min_weight, type_select)
            maskset.save(mask_dir)
        layer_sizes = maskset.layer_sizes
        embedding = maskset.embedding_matrix()
        dbm_tr, dbm_va, dbm_te = (FlyVisibleDataset(b, embedding) for b in (base_tr, base_va, base_te))
        masks, signs = maskset.masks(), maskset.signs()
        print(f"  maskset {maskset.variant}/{maskset.control}: layers {layer_sizes}, couplings {maskset.stats['n_couplings']:,}, "
              f"data-carrying visible units {maskset.stats['n_visible_carrying_data']}/{maskset.n_visible}")
    print(f"  DBM layer sizes: {layer_sizes}")

    batch_size = 128
    train_loader = DataLoader(dbm_tr, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dbm_va, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dbm_te, batch_size=batch_size, shuffle=False)
    cfg = training_config(drive_dir, layer_sizes, dev, epochs_pt, epochs_ft, batch_size, lr_scale)
    print(f"  init={init}, lr_scale={lr_scale} -> lr_pt={cfg['bm']['pretraining_lr']:g}, lr_ft={cfg['bm']['finetuning_lr']:g}")

    # ---------------- DBM training (vendored loops; masked classes injected for fly variants)
    use_masked = (variant != "V0") or force_masked
    ctx = masked_classes(masks, dale=dale, pre_signs=signs if dale else None, init=init) if use_masked else _nullctx()
    t0 = time.time()
    with ctx:
        dbm, _ = pretrain_bm(cfg, train_loader, use_wandb=False, verbose=verbose)
    timings["t_pretrain_s"] = time.time() - t0
    t0 = time.time()
    dbm = joint_finetuning(cfg, dbm, train_loader, val_dataloader=val_loader, use_wandb=False, verbose=verbose)
    timings["t_finetune_s"] = time.time() - t0
    if use_masked:
        n_couplings, n_params = dbm.n_couplings(), dbm.n_params_effective()
    else:
        n_couplings = int(sum(w.numel() for w in dbm.weights))
        n_params = n_couplings + int(sum(b.numel() for b in dbm.biases))
    print(f"  couplings={n_couplings:,} params(eff)={n_params:,}  pretrain {timings['t_pretrain_s']:.0f}s, "
          f"finetune {timings['t_finetune_s']:.0f}s")

    # ---------------- DBM evaluation + belief / drive diagnostics
    dbm_eval = {s: evaluate_dbm(dbm, l, dev, n_iter=10) for s, l in
                [("train", train_loader), ("val", val_loader), ("test", test_loader)]}
    for s, m in dbm_eval.items():
        print(f"  {s:5s}: Free Energy (abs) = {m['free_energy_abs']:.4f}, Recon BCE (mean) = {m['recon_bce']:.4f}")
    diag = belief_stats(dbm, dbm_te, dev)
    diag.update(dbm.drive_stats() if use_masked else dense_drive_stats(dbm))
    print("  diagnostics: " + ", ".join(f"{k}={v:.4g}" for k, v in diag.items()))

    # ---------------- task (i): energy consistency (clamp)
    t0 = time.time()
    energy = {}
    for s, ds in [("train", dbm_tr), ("val", dbm_va), ("test", dbm_te)]:
        print(f"  [energy/{s}]", end="")
        energy[s] = clamp_consistency(dbm, ds, dev, embedding=embedding, n_iter=10, verbose=True)
    timings["t_energy_s"] = time.time() - t0

    # ---------------- baseline MLP on raw 72 features (variant-independent)
    set_seed(seed)
    t0 = time.time()
    baseline = run_baseline_mlp(base_tr, base_va, base_te, dev, hidden_dim=ADAPTER_HIDDEN, n_epochs=adapter_epochs,
                                lr=5e-4, dropout=0.1, batch_size=256, patience=30, forward=0, use_pos_weight=False,
                                save_dir=drive_dir, intervention_column="push", verbose=verbose)
    timings["t_baseline_s"] = time.time() - t0

    # ---------------- tasks (ii)+(iii): adapters on the frozen belief, counterfactual uplift
    set_seed(seed)
    t0 = time.time()
    adapters = run_adapter_pipeline(dbm, dbm_tr, dbm_va, dbm_te, dev, hidden_dim=ADAPTER_HIDDEN,
                                    n_epochs=adapter_epochs, lr=5e-4, dropout=0.1, batch_size=256, save_dir=drive_dir,
                                    patience=30, n_iter=10, use_lrdbm=False, use_wandb=False,
                                    intervention_column="push", forward=0, use_pos_weight=False, verbose=verbose)
    timings["t_adapters_s"] = time.time() - t0
    timings["t_total_s"] = time.time() - t_start

    # ---------------- collect
    belief_dim = int(sum(layer_sizes[1:]))
    best_visit = max(("visit_full", "visit_top"), key=lambda k: adapters[k]["val"]["auc"])
    best_purch = max(("purchase_full", "purchase_top"), key=lambda k: adapters[k]["val"]["auc"])
    et = energy["test"]
    row = {
        "variant": variant, "control": control, "seed": seed, "dale": int(dale), "type_select": type_select,
        "smoke": int(smoke), "run_id": key, "timestamp": datetime.now().isoformat(timespec="seconds"), "device": str(dev),
        "rep": rep, "threads": torch.get_num_threads(), "deterministic": int(deterministic),
        "n_visible": layer_sizes[0], "n_h1": layer_sizes[1], "n_h2": layer_sizes[2],
        "n_h3": layer_sizes[3] if len(layer_sizes) > 3 else 0, "belief_dim": belief_dim,
        "n_couplings": n_couplings, "n_params": n_params,
        "energy_test_delta_mean": et["delta_fe_mean"], "energy_test_frac_pos": et["frac_delta_positive"],
        "energy_test_auc": et["auc_clamped_vs_original"], "energy_test_paired_t": et["paired_t"],
        "energy_test_dfe_high_beta": et.get("delta_fe_mean_high"), "energy_test_dfe_low_beta": et.get("delta_fe_mean_low"),
        "energy_test_welch_t": et.get("welch_t"),
        "baseline_visit_auc": baseline["baseline_visit"]["test"]["auc"],
        "baseline_purchase_auc": baseline["baseline_purchase"]["test"]["auc"],
        "adapter_visit_full_auc": adapters["visit_full"]["test"]["auc"],
        "adapter_visit_top_auc": adapters["visit_top"]["test"]["auc"],
        "adapter_purchase_full_auc": adapters["purchase_full"]["test"]["auc"],
        "adapter_purchase_top_auc": adapters["purchase_top"]["test"]["auc"],
        "best_visit_belief": best_visit.split("_")[1], "best_purchase_belief": best_purch.split("_")[1],
        "cate_push_visit_gamma": adapters["intervention_visit_true_gamma"]["spearman_rho_logit"],
        "cate_push_visit_alpha": adapters["intervention_visit_true_alpha"]["spearman_rho_logit"],
        "cate_push_visit_beta": adapters["intervention_visit_true_beta"]["spearman_rho_logit"],
        "cate_sale1_purchase_alpha": adapters["intervention_purchase_true_alpha"]["spearman_rho_logit"],
        "cate_sale1_purchase_beta": adapters["intervention_purchase_true_beta"]["spearman_rho_logit"],
        "cate_sale1_purchase_gamma": adapters["intervention_purchase_true_gamma"]["spearman_rho_logit"],
        "dbm_test_recon_bce": dbm_eval["test"]["recon_bce"], "dbm_test_free_energy": dbm_eval["test"]["free_energy_abs"],
        "dbm_val_recon_bce_best": dbm_eval["val"]["recon_bce"],
        "init": init, "lr_scale": lr_scale, "tag": tag, "min_weight": min_weight if variant != "V0" else "",
        "n_features_connected": maskset.stats.get("n_features_connected", "") if maskset else 72,
        "belief_sd_median_l1": diag["belief_sd_median_l1"], "belief_sd_median_l2": diag["belief_sd_median_l2"],
        "belief_lowvar_frac_l1": diag["belief_lowvar_frac_l1"], "belief_lowvar_frac_l2": diag["belief_lowvar_frac_l2"],
        "drive_median_l1": diag["drive_median_l1"], "drive_median_l2": diag["drive_median_l2"],
        "energy_test_bias_corr": et["bias_only_corr"], "energy_test_bias_only_mean": et["bias_only_mean"],
        "epochs_pt": epochs_pt, "epochs_ft": epochs_ft, "adapter_epochs": adapter_epochs,
        **{k: round(v, 1) for k, v in timings.items()},
    }
    result = {
        "row": row, "layer_sizes": layer_sizes, "mask_stats": maskset.stats if maskset else None,
        "feature_to_units": maskset.feature_to_units if maskset else None,
        "dbm_eval": dbm_eval, "diagnostics": diag, "energy": energy, "baseline": _jsonable(baseline),
        "adapters": _jsonable(adapters),
        "config": {k: v for k, v in cfg["bm"].items()}, "adapter_config": {
            "hidden_dim": ADAPTER_HIDDEN, "n_epochs": adapter_epochs, "lr": 5e-4, "dropout": 0.1, "batch_size": 256,
            "patience": 30}, "data": ICONIP_DATA, "drive_dir": str(drive_dir),
    }
    runs_dir.mkdir(parents=True, exist_ok=True)
    (runs_dir / f"{key}.json").write_text(json.dumps(result, indent=1, default=_default))
    append_row(results_csv, row)
    print(f"  wrote {runs_dir / (key + '.json')} and a row to {results_csv}")
    print(f"=== done {key}: total {timings['t_total_s']:.0f}s ===")
    return result


@torch.no_grad()
def belief_stats(dbm, dataset, device, n_iter: int = 10, max_rows: int = 30720, batch_size: int = 2048) -> Dict[str, float]:
    """Per-unit standard deviation of the mean-field belief across samples (test split)."""
    dbm.eval(); dbm.to(device)
    n = min(len(dataset), max_rows)
    hs = [[] for _ in range(dbm.n_layers)]
    for i in range(0, n, batch_size):
        v = dataset.data[i:i + batch_size].to(device)
        for l, h in enumerate(dbm.mean_field_inference(v, n_iter=n_iter)):
            hs[l].append(h.cpu())
    out = {}
    for l in range(dbm.n_layers):
        h = torch.cat(hs[l], 0)
        sd = h.std(0)
        out[f"belief_sd_median_l{l + 1}"] = float(sd.median())
        out[f"belief_sd_mean_l{l + 1}"] = float(sd.mean())
        out[f"belief_lowvar_frac_l{l + 1}"] = float((sd < 0.01).float().mean())
        out[f"belief_mean_l{l + 1}"] = float(h.mean())
    return out


@torch.no_grad()
def dense_drive_stats(dbm) -> Dict[str, float]:
    out = {}
    for i, W in enumerate(dbm.weights):
        W = W.detach()
        out[f"drive_median_l{i + 1}"] = float(W.abs().sum(0).median())
        out[f"drive_mean_l{i + 1}"] = float(W.abs().sum(0).mean())
        out[f"fanin_median_l{i + 1}"] = float(W.shape[0])
        out[f"abs_w_mean_l{i + 1}"] = float(W.abs().mean())
    return out


class _nullctx:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


def _default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def _jsonable(d):
    return json.loads(json.dumps(d, default=_default))


def append_row(csv_path: Path, row: Dict[str, Any]):
    """Append (or replace by run_id) one row in results.csv."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if csv_path.exists():
        with open(csv_path, newline="") as f:
            rows = [r for r in csv.DictReader(f) if r.get("run_id") != row["run_id"]]
    rows.append({k: row.get(k, "") for k in CSV_COLUMNS})
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in CSV_COLUMNS})
