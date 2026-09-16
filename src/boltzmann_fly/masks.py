"""Build 0/1 coupling masks for the mushroom-body-constrained DBM.

Variants
--------
V1  : neuron-level, wiring untouched. Visible = all PN neurons of one hemisphere.
      72 PN *types* are chosen (random or top-fanout); each chosen type receives one
      Purchase-World feature, copied to every sister PN of that type. PNs of unchosen
      types are constant 0.
V1b : strict neuron-level. 72 PN *neurons* are chosen at random, one feature each;
      the remaining PNs are constant 0.
V2  : type-level. Visible = 72 chosen PN types (one feature each); PN-type -> KC mask
      is the union over sister PNs.
Hidden layer 1 = KCs with >= 1 PN input at the synapse threshold; hidden layer 2 =
all MBONs of the hemisphere. Masks are symmetrised (edge in either direction).

Controls
--------
degree : degree-preserving random rewiring (Maslov-Sneppen edge swaps, >= 10*E swaps)
         applied independently to each bipartite mask.
er     : Erdos-Renyi random bipartite graph with the same shape and number of edges.

Only text/tables/matrices are produced; nothing here draws or embeds imagery.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .paths import NODES_PARQUET, EDGES_PARQUET, MASK_DIR

# The 72 Purchase-World visible features with ws=4, in the column order produced by the vendored
# simulation generator (see docs/step1_design.md).
PW_FEATURES: List[str] = (
    [f"store_{s}" for s in range(10)]
    + ["loyalty"]
    + [f"age_{d}s" for d in range(20, 70, 10)]
    + [f"income_q{q}" for q in range(1, 6)]
    + [f"month_{m}" for m in range(1, 13)]
    + [f"dow_{d}" for d in range(7)]
    + ["visited_within_7d", "visited_within_14d", "visited_within_30d",
       "purchased_within_7d", "purchased_within_14d", "purchased_within_30d",
       "cum_visits_5plus", "cum_visits_10plus", "cum_visits_30plus",
       "cum_purchases_5plus", "cum_purchases_10plus", "cum_purchases_30plus"]
    + [f"{name}_lag{lag}" for lag in range(1, 5) for name in ["visit", "purchase", "coupon", "campaign", "push"]]
)
assert len(PW_FEATURES) == 72

VARIANTS = ("V1", "V1b", "V2")
CONTROLS = ("none", "degree", "er")


@dataclass
class MaskSet:
    """Everything the model needs, plus bookkeeping for the design note."""
    variant: str
    control: str
    side: str
    min_weight: int
    seed: int
    type_select: str
    # visible layer
    visible_ids: List[str]              # PN bodyIds (V1/V1b) or PN type names (V2)
    visible_sign: np.ndarray            # (n_v,) in {-1, 0, +1}
    feature_to_units: Dict[str, List[int]]  # PW feature -> list of visible unit indices
    # hidden layers
    kc_ids: List[int]
    mbon_ids: List[int]
    mbon_sign: np.ndarray               # (n_mbon,)
    # masks (0/1 float32): (n_v, n_kc), (n_kc, n_mbon)
    mask_pn_kc: np.ndarray
    mask_kc_mbon: np.ndarray
    apl_kc: np.ndarray                  # (n_kc,) 1 if the APL contacts this KC (stored, OFF by default)
    stats: Dict[str, float] = field(default_factory=dict)

    # ---- derived ----
    @property
    def n_visible(self) -> int:
        return len(self.visible_ids)

    @property
    def layer_sizes(self) -> List[int]:
        return [self.n_visible, len(self.kc_ids), len(self.mbon_ids)]

    def embedding_matrix(self) -> np.ndarray:
        """(72, n_visible) 0/1 matrix E: v_fly = v_pw @ E."""
        E = np.zeros((len(PW_FEATURES), self.n_visible), dtype=np.float32)
        for f, units in self.feature_to_units.items():
            E[PW_FEATURES.index(f), units] = 1.0
        return E

    def masks(self) -> List[np.ndarray]:
        return [self.mask_pn_kc, self.mask_kc_mbon]

    def signs(self) -> List[np.ndarray]:
        """Presynaptic sign vectors per coupling layer (Dale option)."""
        return [self.visible_sign.astype(np.float32), np.ones(len(self.kc_ids), dtype=np.float32)]

    # ---- io ----
    def save(self, out_dir: Path = MASK_DIR) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = mask_stem(self.variant, self.control, self.seed, self.side, self.min_weight, self.type_select)
        np.savez_compressed(
            out_dir / f"{stem}.npz",
            mask_pn_kc=self.mask_pn_kc.astype(np.uint8),
            mask_kc_mbon=self.mask_kc_mbon.astype(np.uint8),
            visible_sign=self.visible_sign.astype(np.int8),
            mbon_sign=self.mbon_sign.astype(np.int8),
            apl_kc=self.apl_kc.astype(np.uint8),
        )
        meta = {
            "variant": self.variant, "control": self.control, "side": self.side,
            "min_weight": self.min_weight, "seed": self.seed, "type_select": self.type_select,
            "visible_ids": [str(x) for x in self.visible_ids],
            "feature_to_units": self.feature_to_units,
            "kc_ids": [int(x) for x in self.kc_ids],
            "mbon_ids": [int(x) for x in self.mbon_ids],
            "stats": self.stats,
        }
        (out_dir / f"{stem}.json").write_text(json.dumps(meta, indent=1))
        return out_dir / f"{stem}.npz"

    @classmethod
    def load(cls, variant: str, control: str, seed: int, side: str = "R", min_weight: int = 5,
             type_select: str = "random", mask_dir: Path = MASK_DIR) -> "MaskSet":
        stem = mask_stem(variant, control, seed, side, min_weight, type_select)
        z = np.load(mask_dir / f"{stem}.npz")
        meta = json.loads((mask_dir / f"{stem}.json").read_text())
        return cls(
            variant=meta["variant"], control=meta["control"], side=meta["side"],
            min_weight=meta["min_weight"], seed=meta["seed"], type_select=meta["type_select"],
            visible_ids=meta["visible_ids"], visible_sign=z["visible_sign"].astype(np.int64),
            feature_to_units=meta["feature_to_units"], kc_ids=meta["kc_ids"], mbon_ids=meta["mbon_ids"],
            mbon_sign=z["mbon_sign"].astype(np.int64),
            mask_pn_kc=z["mask_pn_kc"].astype(np.float32), mask_kc_mbon=z["mask_kc_mbon"].astype(np.float32),
            apl_kc=z["apl_kc"].astype(np.float32), stats=meta["stats"],
        )


def mask_stem(variant, control, seed, side="R", min_weight=5, type_select="random") -> str:
    return f"{variant}_{control}_{side}_w{min_weight}_{type_select}_seed{seed}"


# ----------------------------------------------------------------------------- graph helpers
def load_mb(side: str = "R", min_weight: int = 5, nodes_path=NODES_PARQUET, edges_path=EDGES_PARQUET):
    nodes = pd.read_parquet(nodes_path)
    edges = pd.read_parquet(edges_path)
    nodes = nodes[nodes["side"] == side]
    edges = edges[(edges["weight"] >= min_weight) & (edges["side_pre"] == side) & (edges["side_post"] == side)]
    return nodes, edges


def bipartite_mask(edges: pd.DataFrame, rows: List[int], cols: List[int], role_a: str, role_b: str) -> np.ndarray:
    """Symmetrised 0/1 presence matrix between row ids (role_a) and col ids (role_b)."""
    ri = {b: i for i, b in enumerate(rows)}
    ci = {b: j for j, b in enumerate(cols)}
    M = np.zeros((len(rows), len(cols)), dtype=np.float32)
    fwd = edges[(edges.role_pre == role_a) & (edges.role_post == role_b)]
    for a, b in zip(fwd.body_pre.values, fwd.body_post.values):
        if a in ri and b in ci:
            M[ri[a], ci[b]] = 1.0
    bwd = edges[(edges.role_pre == role_b) & (edges.role_post == role_a)]
    for b, a in zip(bwd.body_pre.values, bwd.body_post.values):
        if a in ri and b in ci:
            M[ri[a], ci[b]] = 1.0
    return M


def degree_preserving_rewire(M: np.ndarray, rng: np.random.Generator, n_swaps_per_edge: int = 10,
                             max_tries_factor: int = 100) -> np.ndarray:
    """Maslov-Sneppen edge swaps on a bipartite 0/1 matrix.

    Repeatedly pick two edges (a,b),(c,d) with a != c, b != d and (a,d),(c,b) absent, and
    replace them by (a,d),(c,b). Row and column degrees are preserved exactly. Stops after
    n_swaps_per_edge * E successful swaps (or after max_tries_factor * that many attempts).
    """
    M = M.copy()
    rows, cols = np.nonzero(M)
    E = len(rows)
    edges = np.stack([rows, cols], axis=1)
    present = set(map(tuple, edges.tolist()))
    target = n_swaps_per_edge * E
    done = 0
    tries = 0
    max_tries = max_tries_factor * target
    while done < target and tries < max_tries:
        tries += 1
        i, j = rng.integers(0, E, size=2)
        if i == j:
            continue
        a, b = edges[i]
        c, d = edges[j]
        if a == c or b == d:
            continue
        if (a, d) in present or (c, b) in present:
            continue
        present.discard((a, b)); present.discard((c, d))
        present.add((a, d)); present.add((c, b))
        edges[i, 1] = d
        edges[j, 1] = b
        done += 1
    out = np.zeros_like(M)
    out[edges[:, 0], edges[:, 1]] = 1.0
    assert out.sum() == E
    assert np.array_equal(out.sum(1), M.sum(1)) and np.array_equal(out.sum(0), M.sum(0))
    return out


def erdos_renyi_matched(M: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random bipartite graph with the same shape and the same number of edges."""
    E = int(M.sum())
    flat = rng.choice(M.size, size=E, replace=False)
    out = np.zeros(M.size, dtype=np.float32)
    out[flat] = 1.0
    return out.reshape(M.shape)


# ----------------------------------------------------------------------------- builder
def build_maskset(variant: str, control: str = "none", seed: int = 0, side: str = "R", min_weight: int = 5,
                  type_select: str = "random", nodes_path=NODES_PARQUET, edges_path=EDGES_PARQUET) -> MaskSet:
    assert variant in VARIANTS and control in CONTROLS
    rng = np.random.default_rng(seed)
    nodes, edges = load_mb(side, min_weight, nodes_path, edges_path)

    pn = nodes[nodes.role == "PN"].sort_values("bodyId").reset_index(drop=True)
    kc = nodes[nodes.role == "KC"].sort_values("bodyId").reset_index(drop=True)
    mbon = nodes[nodes.role == "MBON"].sort_values("bodyId").reset_index(drop=True)
    apl = nodes[nodes.role == "APL"]

    pn_ids = pn.bodyId.tolist()
    kc_ids_all = kc.bodyId.tolist()
    mbon_ids = mbon.bodyId.tolist()

    # neuron-level PN-KC presence over all PNs, then restrict hidden-1 to KCs with >= 1 PN input
    M_pn_kc_all = bipartite_mask(edges, pn_ids, kc_ids_all, "PN", "KC")
    kc_keep = np.nonzero(M_pn_kc_all.sum(0) > 0)[0]
    kc_ids = [kc_ids_all[j] for j in kc_keep]
    M_pn_kc_all = M_pn_kc_all[:, kc_keep]
    M_kc_mbon = bipartite_mask(edges, kc_ids, mbon_ids, "KC", "MBON")

    # APL row (stored only; the layered DBM has no within-layer coupling)
    apl_kc = np.zeros(len(kc_ids), dtype=np.float32)
    if len(apl) > 0:
        apl_kc = bipartite_mask(edges, apl.bodyId.tolist(), kc_ids, "APL", "KC")[0]

    # ---- choose 72 PN types (V1, V2) or 72 PN neurons (V1b)
    # 2 right-hemisphere ALPNs have no type annotation; treat each as its own singleton type
    pn_types = [t if isinstance(t, str) else f"untyped_{b}" for t, b in zip(pn["type"].tolist(), pn["bodyId"].tolist())]
    type_names = sorted(set(pn_types))
    n_feat = len(PW_FEATURES)
    # PNs / types that actually contact a KC at this threshold (a feature placed elsewhere never enters the model)
    pn_projecting = M_pn_kc_all.sum(1) > 0
    projecting_types = sorted({t for t, ok in zip(pn_types, pn_projecting) if ok})
    if variant in ("V1", "V2"):
        if type_select == "random":
            chosen_types = list(rng.choice(type_names, size=n_feat, replace=False))
        elif type_select == "random-projecting":
            if len(projecting_types) < n_feat:
                raise ValueError(f"only {len(projecting_types)} PN types project to a KC at w>={min_weight}; need {n_feat}")
            chosen_types = list(rng.choice(projecting_types, size=n_feat, replace=False))
        elif type_select == "top-fanout":
            fan = {}
            for t in type_names:
                idx = [i for i, tt in enumerate(pn_types) if tt == t]
                fan[t] = int((M_pn_kc_all[idx].sum(0) > 0).sum())  # distinct KC targets of the type
            chosen_types = sorted(type_names, key=lambda t: (-fan[t], t))[:n_feat]
        else:
            raise ValueError(type_select)
        perm = rng.permutation(n_feat)  # feature f -> chosen_types[perm[f]]
        feature_to_type = {PW_FEATURES[f]: chosen_types[perm[f]] for f in range(n_feat)}
    else:
        chosen_types = None
        feature_to_type = None

    if variant == "V1":
        visible_ids = [str(b) for b in pn_ids]
        visible_sign = pn["sign"].to_numpy()
        type_to_units = {}
        for i, t in enumerate(pn_types):
            type_to_units.setdefault(t, []).append(i)
        feature_to_units = {f: type_to_units[t] for f, t in feature_to_type.items()}
        mask_pn_kc = M_pn_kc_all
    elif variant == "V1b":
        visible_ids = [str(b) for b in pn_ids]
        visible_sign = pn["sign"].to_numpy()
        if type_select == "random":
            pool = np.arange(len(pn_ids))
        elif type_select == "random-projecting":
            pool = np.nonzero(pn_projecting)[0]
        else:
            raise ValueError(f"V1b supports type_select random | random-projecting, got {type_select}")
        if len(pool) < n_feat:
            raise ValueError(f"only {len(pool)} PNs project to a KC at w>={min_weight}; need {n_feat}")
        chosen_units = rng.choice(pool, size=n_feat, replace=False)
        feature_to_units = {PW_FEATURES[f]: [int(chosen_units[f])] for f in range(n_feat)}
        mask_pn_kc = M_pn_kc_all
    else:  # V2: type-level visible layer
        visible_types = [feature_to_type[f] for f in PW_FEATURES]  # unit f carries feature f
        visible_ids = visible_types
        mask_pn_kc = np.zeros((n_feat, len(kc_ids)), dtype=np.float32)
        visible_sign = np.zeros(n_feat, dtype=np.int64)
        for f, t in enumerate(visible_types):
            idx = [i for i, tt in enumerate(pn_types) if tt == t]
            mask_pn_kc[f] = (M_pn_kc_all[idx].sum(0) > 0).astype(np.float32)
            s = pn["sign"].to_numpy()[idx]
            visible_sign[f] = int(np.sign(s.sum())) if (s != 0).any() else 0
        feature_to_units = {PW_FEATURES[f]: [f] for f in range(n_feat)}

    # ---- controls
    real_pn_kc, real_kc_mbon = mask_pn_kc, M_kc_mbon
    if control == "degree":
        mask_pn_kc = degree_preserving_rewire(real_pn_kc, rng)
        M_kc_mbon = degree_preserving_rewire(real_kc_mbon, rng)
    elif control == "er":
        mask_pn_kc = erdos_renyi_matched(real_pn_kc, rng)
        M_kc_mbon = erdos_renyi_matched(real_kc_mbon, rng)

    # ---- stats
    carrying = sorted({u for units in feature_to_units.values() for u in units})
    n_v = len(visible_ids)
    kc_with_data = int((mask_pn_kc[carrying].sum(0) > 0).sum())
    Emb = np.zeros((n_feat, n_v), dtype=np.float32)
    for f, units in feature_to_units.items():
        Emb[PW_FEATURES.index(f), units] = 1.0
    features_connected = int((((Emb @ mask_pn_kc) > 0).sum(1) > 0).sum())
    stats = {
        "n_visible": n_v,
        "n_visible_carrying_data": len(carrying),
        "n_kc": len(kc_ids),
        "n_kc_dropped_no_pn_input": len(kc_ids_all) - len(kc_ids),
        "n_kc_reached_by_data_units": kc_with_data,
        "n_kc_with_mbon_output": int((M_kc_mbon.sum(1) > 0).sum()),
        "n_mbon": len(mbon_ids),
        "edges_pn_kc": int(mask_pn_kc.sum()),
        "edges_kc_mbon": int(M_kc_mbon.sum()),
        "density_pn_kc": float(mask_pn_kc.mean()),
        "density_kc_mbon": float(M_kc_mbon.mean()),
        "n_couplings": int(mask_pn_kc.sum() + M_kc_mbon.sum()),
        "n_params_total": int(mask_pn_kc.sum() + M_kc_mbon.sum() + n_v + len(kc_ids) + len(mbon_ids)),
        "overlap_with_real_pn_kc": float((mask_pn_kc * real_pn_kc).sum() / max(real_pn_kc.sum(), 1)),
        "overlap_with_real_kc_mbon": float((M_kc_mbon * real_kc_mbon).sum() / max(real_kc_mbon.sum(), 1)),
        "n_pn_types": len(type_names),
        "n_pn_projecting": int(pn_projecting.sum()),
        "n_pn_types_projecting": len(projecting_types),
        "n_features_connected": features_connected,
        "chosen_types": chosen_types,
        "apl_kc_contacts": int(apl_kc.sum()),
    }
    return MaskSet(
        variant=variant, control=control, side=side, min_weight=min_weight, seed=seed, type_select=type_select,
        visible_ids=visible_ids, visible_sign=np.asarray(visible_sign, dtype=np.int64),
        feature_to_units=feature_to_units, kc_ids=[int(x) for x in kc_ids], mbon_ids=[int(x) for x in mbon_ids],
        mbon_sign=mbon["sign"].to_numpy().astype(np.int64),
        mask_pn_kc=mask_pn_kc.astype(np.float32), mask_kc_mbon=M_kc_mbon.astype(np.float32),
        apl_kc=apl_kc.astype(np.float32), stats=stats,
    )
