"""Mask construction, rewiring controls, and feature embedding on a synthetic mini-connectome."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from boltzmann_fly.masks import (PW_FEATURES, build_maskset, degree_preserving_rewire, erdos_renyi_matched, MaskSet)


def test_rewire_preserves_degrees_and_changes_edges():
    rng = np.random.default_rng(0)
    M = (rng.random((60, 80)) < 0.1).astype(np.float32)
    R = degree_preserving_rewire(M, np.random.default_rng(1))
    assert R.sum() == M.sum()
    assert np.array_equal(R.sum(0), M.sum(0)) and np.array_equal(R.sum(1), M.sum(1))
    assert (R * M).sum() < M.sum()  # actually rewired
    E = erdos_renyi_matched(M, np.random.default_rng(2))
    assert E.sum() == M.sum() and E.shape == M.shape


def _synthetic(tmp_path: Path, n_pn=120, n_types=90, n_kc=300, n_mbon=7, seed=0):
    rng = np.random.default_rng(seed)
    body = 1000
    rows = []
    types = [f"T{i}" for i in range(n_types)]
    pn_ids, kc_ids, mbon_ids = [], [], []
    for i in range(n_pn):
        rows.append(dict(bodyId=body, role="PN", side="R", type=types[i % n_types], sign=int(rng.choice([1, -1])))); pn_ids.append(body); body += 1
    for i in range(n_kc):
        rows.append(dict(bodyId=body, role="KC", side="R", type="KCg-m", sign=1)); kc_ids.append(body); body += 1
    for i in range(n_mbon):
        rows.append(dict(bodyId=body, role="MBON", side="R", type=f"MBON{i:02d}", sign=int(rng.choice([1, -1])))); mbon_ids.append(body); body += 1
    rows.append(dict(bodyId=body, role="APL", side="R", type="APL", sign=-1)); apl = body; body += 1
    rows.append(dict(bodyId=body, role="PN", side="L", type="T0", sign=1))  # other side, must be ignored
    nodes = pd.DataFrame(rows)
    edges = []
    for p in pn_ids:
        for k in rng.choice(kc_ids, size=6, replace=False):
            edges.append(dict(body_pre=p, body_post=int(k), weight=int(rng.integers(1, 20)), role_pre="PN", role_post="KC", side_pre="R", side_post="R", sign_pre=1))
    for k in kc_ids[: n_kc - 10]:  # 10 KCs get no MBON output; all KCs above have PN input unless dropped below
        for m in rng.choice(mbon_ids, size=2, replace=False):
            edges.append(dict(body_pre=k, body_post=int(m), weight=int(rng.integers(1, 20)), role_pre="KC", role_post="MBON", side_pre="R", side_post="R", sign_pre=1))
    for k in kc_ids:
        edges.append(dict(body_pre=apl, body_post=k, weight=30, role_pre="APL", role_post="KC", side_pre="R", side_post="R", sign_pre=-1))
    edges = pd.DataFrame(edges)
    nodes.to_parquet(tmp_path / "n.parquet"); edges.to_parquet(tmp_path / "e.parquet")
    return tmp_path / "n.parquet", tmp_path / "e.parquet"


@pytest.mark.parametrize("variant", ["V1", "V1b", "V2"])
@pytest.mark.parametrize("control", ["none", "degree", "er"])
def test_build_variants(tmp_path, variant, control):
    npath, epath = _synthetic(tmp_path)
    ms = build_maskset(variant, control, seed=0, side="R", min_weight=5, nodes_path=npath, edges_path=epath)
    E = ms.embedding_matrix()
    assert E.shape == (72, ms.n_visible)
    assert (E.sum(0) <= 1).all()  # a visible unit carries at most one feature
    assert (E.sum(1) >= 1).all()  # every feature is carried by >= 1 unit
    if variant == "V1b":
        assert E.sum() == 72 and ms.n_visible == 120
    if variant == "V2":
        assert ms.n_visible == 72 and np.array_equal(E, np.eye(72, dtype=np.float32))
    if variant == "V1":
        assert ms.n_visible == 120 and E.sum() >= 72
    assert ms.mask_pn_kc.shape == (ms.n_visible, len(ms.kc_ids))
    assert ms.mask_kc_mbon.shape == (len(ms.kc_ids), 7)
    assert ms.stats["n_couplings"] == ms.mask_pn_kc.sum() + ms.mask_kc_mbon.sum()
    assert ms.apl_kc.sum() == len(ms.kc_ids)
    # round trip
    ms.save(tmp_path / "masks")
    ms2 = MaskSet.load(variant, control, 0, "R", 5, "random", tmp_path / "masks")
    assert np.array_equal(ms2.mask_pn_kc, ms.mask_pn_kc) and ms2.feature_to_units == ms.feature_to_units
    if control == "degree":
        real = build_maskset(variant, "none", seed=0, nodes_path=npath, edges_path=epath)
        assert np.array_equal(real.mask_pn_kc.sum(1), ms.mask_pn_kc.sum(1))
        assert np.array_equal(real.mask_kc_mbon.sum(0), ms.mask_kc_mbon.sum(0))
        assert real.feature_to_units == ms.feature_to_units  # same seed -> same feature assignment
    if control == "er":
        real = build_maskset(variant, "none", seed=0, nodes_path=npath, edges_path=epath)
        assert real.mask_pn_kc.sum() == ms.mask_pn_kc.sum()


def test_top_fanout_selection(tmp_path):
    npath, epath = _synthetic(tmp_path)
    ms = build_maskset("V2", "none", seed=0, type_select="top-fanout", nodes_path=npath, edges_path=epath)
    assert ms.n_visible == 72 and len(set(ms.visible_ids)) == 72
