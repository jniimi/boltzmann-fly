"""Task (i): energy-based consistency evaluation (ICONIP Sec. 4.1).

Eligible samples: purchase_lag1..4 == 0. Clamp: purchase_lag1..4 = 1, purchased_within_7d = 1,
campaign_lag1..4 = 0, push_lag1..4 = 0. Delta F = F(clamped) - F(original), mean-field n_iter=10.
Reported per split: paired t / Wilcoxon against zero, median split on true_beta (Welch t,
Mann-Whitney U), plus a scalar summary AUC(F separates clamped from original vectors).
The clamp is defined on the 72 Purchase-World features and then embedded into the PN layer.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from scipy import stats
from sklearn.metrics import roc_auc_score

from .data import SimulationDataset

CONDITION = {f"purchase_lag{i}": 0 for i in range(1, 5)}
CLAMP = {**{f"purchase_lag{i}": 1 for i in range(1, 5)}, "purchased_within_7d": 1,
         **{f"campaign_lag{i}": 0 for i in range(1, 5)}, **{f"push_lag{i}": 0 for i in range(1, 5)}}


@torch.no_grad()
def free_energies(dbm, v: torch.Tensor, device, n_iter=10, batch_size=2048) -> np.ndarray:
    dbm.eval(); dbm.to(device)
    out = []
    for i in range(0, len(v), batch_size):
        out.append(dbm.free_energy(v[i:i + batch_size].to(device), n_iter=n_iter).cpu())
    return torch.cat(out).numpy()


def clamp_consistency(dbm, dataset: SimulationDataset, device, embedding: np.ndarray | None = None,
                      n_iter: int = 10, split_by: str = "true_beta", verbose: bool = True) -> Dict[str, float]:
    fmap = dataset.get_feature_index_map()
    v_pw = dataset.pw_data if hasattr(dataset, "pw_data") else dataset.data  # (N, 72)
    mask = torch.ones(len(v_pw), dtype=torch.bool)
    for col, val in CONDITION.items():
        mask &= v_pw[:, fmap[col]] == val
    idx = torch.nonzero(mask).squeeze(1)
    v_orig = v_pw[idx].clone()
    v_clamp = v_orig.clone()
    for col, val in CLAMP.items():
        v_clamp[:, fmap[col]] = float(val)
    if embedding is not None:
        E = torch.as_tensor(embedding, dtype=torch.float32)
        v_orig, v_clamp = v_orig @ E, v_clamp @ E
    fe_o = free_energies(dbm, v_orig, device, n_iter)
    fe_c = free_energies(dbm, v_clamp, device, n_iter)
    d = fe_c - fe_o
    # visible-bias-only part of dF: -(v_clamped - v_orig) . b_v  (context-free marginal penalty)
    with torch.no_grad():
        b_v = dbm.biases[0].detach().cpu().float()
        d_bias = (-(v_clamp - v_orig) @ b_v).numpy()
    bias_corr = float(np.corrcoef(d, d_bias)[0, 1]) if d.std() > 0 and d_bias.std() > 0 else float("nan")
    t, tp = stats.ttest_rel(fe_c, fe_o)
    w, wp = stats.wilcoxon(d)
    auc = float(roc_auc_score(np.r_[np.zeros(len(fe_o)), np.ones(len(fe_c))], np.r_[fe_o, fe_c]))
    res = {
        "n_eligible": int(len(idx)), "delta_fe_mean": float(d.mean()), "delta_fe_std": float(d.std(ddof=1)),
        "frac_delta_positive": float((d > 0).mean()), "auc_clamped_vs_original": auc,
        "paired_t": float(t), "paired_t_p": float(tp), "wilcoxon_p": float(wp),
        "fe_original_mean": float(fe_o.mean()), "fe_clamped_mean": float(fe_c.mean()),
        "bias_only_mean": float(d_bias.mean()), "bias_only_corr": bias_corr,
        "coupling_part_std": float((d - d_bias).std(ddof=1)),
    }
    if split_by in dataset.df.columns:
        b = dataset.df[split_by].values[idx.numpy()]
        med = float(np.median(b))
        hi, lo = d[b >= med], d[b < med]
        wt, wtp = stats.ttest_ind(hi, lo, equal_var=False)
        u, up = stats.mannwhitneyu(hi, lo, alternative="two-sided")
        res.update({
            "split_by": split_by, "split_median": med, "n_high": int(len(hi)), "n_low": int(len(lo)),
            "delta_fe_mean_high": float(hi.mean()), "delta_fe_mean_low": float(lo.mean()),
            "delta_fe_std_high": float(hi.std(ddof=1)), "delta_fe_std_low": float(lo.std(ddof=1)),
            "welch_t": float(wt), "welch_p": float(wtp), "mann_whitney_p": float(up),
        })
    if verbose:
        print(f"  clamp consistency: n={res['n_eligible']}, dF mean={res['delta_fe_mean']:+.4f} "
              f"(sd {res['delta_fe_std']:.3f}), frac>0={res['frac_delta_positive']:.3f}, "
              f"AUC={auc:.4f}, paired t={t:.2f}, corr(dF, bias-only)={bias_corr:.3f}"
              + (f", high/low beta dF={res['delta_fe_mean_high']:+.3f}/{res['delta_fe_mean_low']:+.3f}, "
                 f"Welch t={res['welch_t']:.2f}" if "welch_t" in res else ""))
    return res
