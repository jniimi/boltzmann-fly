#!/usr/bin/env python
"""Step 0: extract the mushroom-body (MB) subgraph from the MaleCNS v1.0 flat connectome.

Node set  : status == 'Traced' and (class in {ALPN, Kenyon_Cell, MBON, DAN} or type in {APL, DPM}).
Roles     : PN (class ALPN), KC (Kenyon_Cell), MBON, DAN, APL (type), DPM (type).
Edges     : directed pre -> post, weight = synapse count (min confidence 0.5, traced-only file).
Signs     : consensus_nt, falling back to celltype_predicted_nt when consensus is 'unclear'.
            acetylcholine -> +1, gaba -> -1, glutamate -> -1, everything else / unclear -> 0 (unknown).
Thresholds: weight >= 1 (all), >= 5 ("significant"), >= 10.

Outputs (out-dir):  mb_nodes.parquet, mb_edges.parquet, mb_stats.json
Report   (repo)  :  docs/step0_mb_stats.md   (text and tables only; no imagery)

The script is idempotent: every run overwrites the same output files.
Run with:  uv run python scripts/step0_extract_mb.py [--data-dir ...] [--out-dir ...]
"""
from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as pf
import scipy.sparse as sp

# ----------------------------------------------------------------------------- constants
DATA_DIR_DEFAULT = Path("/Volumes/EXTERNAL/malecns/v1.0/flat-connectome")
OUT_DIR_DEFAULT = Path("/Volumes/EXTERNAL/malecns/derived/boltzmann-fly")
REPO_DIR_DEFAULT = Path(__file__).resolve().parent.parent

F_EDGES = "connectome-weights-male-cns-v1.0-minconf-0.5-traced-only.feather"
F_ANNOT = "body-annotations-male-cns-v1.0-minconf-0.5.feather"
F_NT = "body-neurotransmitters-male-cns-v1.0.feather"

CLASS_TO_ROLE = {"ALPN": "PN", "Kenyon_Cell": "KC", "MBON": "MBON", "DAN": "DAN"}
TYPE_TO_ROLE = {"APL": "APL", "DPM": "DPM"}
ROLES = ["PN", "KC", "MBON", "DAN", "APL", "DPM"]
SIDES = ["L", "R", "M", "NaN"]
THRESHOLDS = [1, 5, 10]
NT_SIGN = {"acetylcholine": 1, "gaba": -1, "glutamate": -1}
CORE_PAIRS = [("PN", "KC"), ("KC", "KC"), ("KC", "MBON"), ("MBON", "KC"),
              ("APL", "KC"), ("KC", "APL"), ("DAN", "KC"), ("DAN", "MBON")]
QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]
# Purchase-World reference DBM (ICONIP 2026 paper): visible 72, hidden 64-32-16.
PW_LAYERS = [72, 64, 32, 16]


# ----------------------------------------------------------------------------- helpers
class Timer:
    def __init__(self):
        self.t0 = time.perf_counter()
        self.marks: dict[str, float] = {}

    def mark(self, name: str):
        t = time.perf_counter()
        self.marks[name] = round(t - self.t0, 2)
        self.t0 = t
        print(f"[{name}] {self.marks[name]:.2f}s  peak_rss={peak_rss_gb():.2f} GB", flush=True)


def peak_rss_gb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux kilobytes
    return rss / 1e9 if sys.platform == "darwin" else rss / 1e6


def side_label(s) -> str:
    return "NaN" if (s is None or (isinstance(s, float) and np.isnan(s))) else str(s)


def role_pair_table(edges: pd.DataFrame, value: str | None) -> pd.DataFrame:
    """6x6 table role_pre (rows) x role_post (cols). value=None -> edge count, else sum."""
    if value is None:
        t = edges.groupby(["role_pre", "role_post"], observed=True).size()
    else:
        t = edges.groupby(["role_pre", "role_post"], observed=True)[value].sum()
    return t.unstack("role_post").reindex(index=ROLES, columns=ROLES).fillna(0).astype(int)


def md_table(df: pd.DataFrame, index_name: str = "") -> str:
    df = df.copy()
    cols = [index_name] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for idx, row in df.iterrows():
        cells = [str(idx)] + [fmt(v) for v in row.values]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def fmt(v) -> str:
    if isinstance(v, (bool, np.bool_)):
        return str(v)
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}"
    if isinstance(v, (float, np.floating)):
        if np.isnan(v):
            return "nan"
        if float(v).is_integer() and abs(v) < 1e15:
            return f"{int(v):,}"
        return f"{v:.4g}" if abs(v) < 1e-2 or abs(v) >= 1e6 else f"{v:,.3f}"
    return str(v)


def quantile_summary(x: pd.Series) -> dict:
    x = x.astype(float)
    d = {"n": int(len(x)), "mean": float(x.mean()) if len(x) else float("nan"),
         "std": float(x.std()) if len(x) > 1 else float("nan")}
    for q in QUANTILES:
        d[f"q{int(q*100):02d}"] = float(x.quantile(q)) if len(x) else float("nan")
    d["min"] = float(x.min()) if len(x) else float("nan")
    d["max"] = float(x.max()) if len(x) else float("nan")
    return d


def to_jsonable(o):
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, pd.DataFrame):
        return {str(i): {str(c): to_jsonable(v) for c, v in r.items()} for i, r in o.iterrows()}
    if isinstance(o, pd.Series):
        return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, float) and np.isnan(o):
        return None
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


# ----------------------------------------------------------------------------- main steps
def load_nodes(data_dir: Path) -> tuple[pd.DataFrame, dict]:
    cols = ["bodyId", "type", "class", "superclass", "somaSide", "status", "instance", "group"]
    ann = pf.read_table(data_dir / F_ANNOT, columns=cols).to_pandas()
    traced = ann["status"] == "Traced"
    in_class = ann["class"].isin(CLASS_TO_ROLE)
    in_type = ann["type"].isin(TYPE_TO_ROLE)
    nodes = ann[traced & (in_class | in_type)].copy()
    nodes["role"] = nodes["class"].map(CLASS_TO_ROLE)
    nodes.loc[nodes["type"].isin(TYPE_TO_ROLE), "role"] = nodes["type"].map(TYPE_TO_ROLE)
    nodes["side"] = nodes["somaSide"].map(side_label)
    nodes = nodes.sort_values("bodyId").reset_index(drop=True)
    extra = {
        "annotation_rows_total": int(len(ann)),
        "status_traced_total": int(traced.sum()),
        "not_traced_in_selected_classes_or_types": int(((in_class | in_type) & ~traced).sum()),
        "apl_dpm_class_values": {t: sorted(map(str, nodes.loc[nodes["type"] == t, "class"].fillna("NaN").unique()))
                                 for t in TYPE_TO_ROLE},
        "apl_dpm_superclass_values": {t: sorted(map(str, nodes.loc[nodes["type"] == t, "superclass"].fillna("NaN").unique()))
                                      for t in TYPE_TO_ROLE},
    }
    return nodes, extra


def attach_signs(nodes: pd.DataFrame, data_dir: Path) -> tuple[pd.DataFrame, dict]:
    nt = pf.read_table(data_dir / F_NT, columns=["body", "consensus_nt", "celltype_predicted_nt",
                                                 "predicted_nt", "predicted_nt_confidence"]).to_pandas()
    nt = nt.rename(columns={"body": "bodyId"})
    nodes = nodes.merge(nt, on="bodyId", how="left")
    cons = nodes["consensus_nt"].fillna("unclear")
    ctp = nodes["celltype_predicted_nt"].fillna("unclear")
    use_fallback = (cons == "unclear") & (ctp != "unclear")
    nodes["nt"] = np.where(cons != "unclear", cons, np.where(use_fallback, ctp, "unclear"))
    nodes["nt_source"] = np.where(cons != "unclear", "consensus",
                                  np.where(use_fallback, "celltype_predicted", "none"))
    nodes["sign"] = nodes["nt"].map(NT_SIGN).fillna(0).astype(int)
    info = {
        "nodes_without_nt_row": int(nodes["consensus_nt"].isna().sum()),
        "sign_from_consensus": int((nodes["nt_source"] == "consensus").sum()),
        "sign_from_celltype_fallback": int((nodes["nt_source"] == "celltype_predicted").sum()),
        "nt_unclear_after_fallback": int((nodes["nt"] == "unclear").sum()),
        "fallback_by_role": nodes[use_fallback].groupby("role").size().reindex(ROLES).fillna(0).astype(int).to_dict(),
        "sign_unknown_by_role": nodes[nodes["sign"] == 0].groupby("role").size().reindex(ROLES).fillna(0).astype(int).to_dict(),
    }
    return nodes, info


def load_edges(nodes: pd.DataFrame, data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    ids = pa.array(nodes["bodyId"].to_numpy(), type=pa.int64())
    tbl = pf.read_table(data_dir / F_EDGES, columns=["body_pre", "body_post", "weight"])
    n_total = tbl.num_rows
    total_syn = pc.sum(tbl["weight"]).as_py()
    pre_in = pc.is_in(tbl["body_pre"], value_set=ids)
    post_in = pc.is_in(tbl["body_post"], value_set=ids)
    both = pc.and_(pre_in, post_in)
    induced = tbl.filter(both).to_pandas()

    # boundary edges: exactly one endpoint in the set
    out_tbl = tbl.filter(pc.and_(pre_in, pc.invert(post_in))).select(["body_pre", "weight"]).to_pandas()
    in_tbl = tbl.filter(pc.and_(post_in, pc.invert(pre_in))).select(["body_post", "weight"]).to_pandas()
    del tbl, pre_in, post_in, both
    out_agg = out_tbl.groupby("body_pre")["weight"].agg(ext_out_edges="size", ext_out_syn="sum")
    in_agg = in_tbl.groupby("body_post")["weight"].agg(ext_in_edges="size", ext_in_syn="sum")
    # internal degrees
    int_out = induced.groupby("body_pre")["weight"].agg(int_out_edges="size", int_out_syn="sum")
    int_in = induced.groupby("body_post")["weight"].agg(int_in_edges="size", int_in_syn="sum")
    bnd = pd.DataFrame(index=nodes["bodyId"].to_numpy())
    for a in (out_agg, in_agg, int_out, int_in):
        bnd = bnd.join(a, how="left")
    bnd = bnd.fillna(0).astype(int)
    bnd.index.name = "bodyId"
    bnd = bnd.reset_index()

    # decorate induced edges
    meta = nodes.set_index("bodyId")[["role", "side", "sign"]]
    induced["role_pre"] = induced["body_pre"].map(meta["role"]).astype("category")
    induced["role_post"] = induced["body_post"].map(meta["role"]).astype("category")
    induced["side_pre"] = induced["body_pre"].map(meta["side"])
    induced["side_post"] = induced["body_post"].map(meta["side"])
    induced["sign_pre"] = induced["body_pre"].map(meta["sign"]).astype(int)
    induced = induced.sort_values(["body_pre", "body_post"]).reset_index(drop=True)
    info = {
        "edge_rows_total": int(n_total),
        "synapses_total": int(total_syn),
        "induced_edges": int(len(induced)),
        "induced_synapses": int(induced["weight"].sum()),
        "boundary_out_edges": int(len(out_tbl)),
        "boundary_out_synapses": int(out_tbl["weight"].sum()),
        "boundary_in_edges": int(len(in_tbl)),
        "boundary_in_synapses": int(in_tbl["weight"].sum()),
        "duplicate_pre_post_rows": int(induced.duplicated(["body_pre", "body_post"]).sum()),
        "self_loops": int((induced["body_pre"] == induced["body_post"]).sum()),
    }
    return induced, bnd, info


def boundary_by_role(nodes: pd.DataFrame, bnd: pd.DataFrame) -> pd.DataFrame:
    b = bnd.merge(nodes[["bodyId", "role"]], on="bodyId")
    g = b.groupby("role")[["int_in_syn", "ext_in_syn", "int_out_syn", "ext_out_syn",
                            "int_in_edges", "ext_in_edges", "int_out_edges", "ext_out_edges"]].sum()
    g = g.reindex(ROLES).fillna(0).astype(int)
    g["frac_in_syn_kept"] = g["int_in_syn"] / (g["int_in_syn"] + g["ext_in_syn"]).replace(0, np.nan)
    g["frac_out_syn_kept"] = g["int_out_syn"] / (g["int_out_syn"] + g["ext_out_syn"]).replace(0, np.nan)
    return g


def per_side_core(edges: pd.DataFrame, nodes: pd.DataFrame) -> dict:
    """Core role pairs per side (both endpoints on the same side), at each threshold."""
    out = {}
    for side in ["L", "R"]:
        e = edges[(edges["side_pre"] == side) & (edges["side_post"] == side)]
        rows = {}
        for a, b in CORE_PAIRS:
            ee = e[(e["role_pre"] == a) & (e["role_post"] == b)]
            r = {}
            for th in THRESHOLDS:
                et = ee[ee["weight"] >= th]
                r[f"edges_w{th}"] = int(len(et))
                r[f"syn_w{th}"] = int(et["weight"].sum())
            rows[f"{a}->{b}"] = r
        out[side] = rows
    cross = edges[edges["side_pre"] != edges["side_post"]]
    out["cross_side_edges"] = int(len(cross))
    out["cross_side_synapses"] = int(cross["weight"].sum())
    out["cross_side_by_pair"] = role_pair_table(cross, None)
    return out


def kc_indegree(edges: pd.DataFrame, nodes: pd.DataFrame) -> dict:
    kc_ids = nodes.loc[nodes["role"] == "KC", ["bodyId", "side"]]
    pk = edges[(edges["role_pre"] == "PN") & (edges["role_post"] == "KC")]
    out = {}
    for th in THRESHOLDS:
        e = pk[pk["weight"] >= th]
        deg = e.groupby("body_post")["body_pre"].nunique()
        deg = deg.reindex(kc_ids["bodyId"]).fillna(0)  # KCs with no PN input at this threshold count as 0
        d = {"all": quantile_summary(deg)}
        d["all"]["n_kc_zero"] = int((deg == 0).sum())
        for side in ["L", "R"]:
            ids = kc_ids.loc[kc_ids["side"] == side, "bodyId"]
            d[side] = quantile_summary(deg.reindex(ids).fillna(0))
        # excluding zero-degree KCs
        d["all_nonzero"] = quantile_summary(deg[deg > 0])
        out[f"w{th}"] = d
    return out


def densities(edges: pd.DataFrame, nodes: pd.DataFrame) -> dict:
    out = {}
    for side in ["L", "R"]:
        n = nodes[nodes["side"] == side].groupby("role").size().reindex(ROLES).fillna(0).astype(int)
        e = edges[(edges["side_pre"] == side) & (edges["side_post"] == side)]
        d = {"n_PN": int(n["PN"]), "n_KC": int(n["KC"]), "n_MBON": int(n["MBON"])}
        for a, b in [("PN", "KC"), ("KC", "MBON"), ("MBON", "KC"), ("KC", "PN")]:
            ee = e[(e["role_pre"] == a) & (e["role_post"] == b)]
            for th in THRESHOLDS:
                cnt = int((ee["weight"] >= th).sum())
                d[f"{a}->{b}_edges_w{th}"] = cnt
                d[f"{a}->{b}_density_w{th}"] = cnt / (n[a] * n[b]) if n[a] * n[b] else float("nan")
        # undirected union masks (what a BM would use)
        for a, b in [("PN", "KC"), ("KC", "MBON")]:
            fwd = e[(e["role_pre"] == a) & (e["role_post"] == b)]
            bwd = e[(e["role_pre"] == b) & (e["role_post"] == a)]
            for th in THRESHOLDS:
                pairs = set(map(tuple, fwd.loc[fwd["weight"] >= th, ["body_pre", "body_post"]].to_numpy())) | \
                        set(map(tuple, bwd.loc[bwd["weight"] >= th, ["body_post", "body_pre"]].to_numpy()))
                d[f"{a}-{b}_union_pairs_w{th}"] = len(pairs)
                d[f"{a}-{b}_union_density_w{th}"] = len(pairs) / (n[a] * n[b]) if n[a] * n[b] else float("nan")
        out[side] = d
    return out


def reciprocity(edges: pd.DataFrame) -> dict:
    key = edges["body_pre"].astype(np.int64) * (1 << 32) + edges["body_post"].astype(np.int64)
    rkey = edges["body_post"].astype(np.int64) * (1 << 32) + edges["body_pre"].astype(np.int64)
    w_by_key = pd.Series(edges["weight"].to_numpy(), index=key.to_numpy())
    rev_w = w_by_key.reindex(rkey.to_numpy()).to_numpy()  # NaN if reverse edge absent
    has_rev = ~np.isnan(rev_w)
    e = edges.assign(has_rev=has_rev, rev_w=np.nan_to_num(rev_w, nan=0.0))
    out = {}
    for name, mask in [("any", np.ones(len(e), bool)), ("both_ge5", (e["weight"] >= 5).to_numpy())]:
        sub = e[mask]
        if name == "both_ge5":
            ok = sub["has_rev"] & (sub["rev_w"] >= 5)
        else:
            ok = sub["has_rev"]
        overall = float(ok.mean()) if len(sub) else float("nan")
        per = sub.assign(ok=ok).groupby(["role_pre", "role_post"], observed=True)["ok"].mean() \
                 .unstack("role_post").reindex(index=ROLES, columns=ROLES)
        out[name] = {"overall": overall, "n_edges": int(len(sub)), "n_reciprocated": int(ok.sum()), "per_pair": per}
    return out


def sym_antisym(edges: pd.DataFrame, nodes: pd.DataFrame) -> dict:
    known = nodes[nodes["sign"] != 0]
    idx = pd.Series(np.arange(len(known)), index=known["bodyId"].to_numpy())
    e = edges[edges["body_pre"].isin(idx.index) & edges["body_post"].isin(idx.index)]
    e = e[e["body_pre"] != e["body_post"]]
    r = idx.reindex(e["body_pre"]).to_numpy()
    c = idx.reindex(e["body_post"]).to_numpy()
    n = len(known)
    out = {"n_nodes_with_sign": int(n), "n_nodes_excluded_unknown_sign": int(len(nodes) - n),
           "n_edges_used": int(len(e)), "n_edges_dropped_unknown_sign": int(len(edges) - len(e))}
    for name, w in [("raw", e["weight"].to_numpy(float)), ("log1p", np.log1p(e["weight"].to_numpy(float)))]:
        vals = e["sign_pre"].to_numpy(float) * w
        W = sp.csr_matrix((vals, (r, c)), shape=(n, n))
        S = (W + W.T) * 0.5
        A = (W - W.T) * 0.5
        nS = sp.linalg.norm(S, "fro")
        nA = sp.linalg.norm(A, "fro")
        nW = sp.linalg.norm(W, "fro")
        out[name] = {"fro_W": float(nW), "fro_S": float(nS), "fro_A": float(nA),
                     "ratio_A_over_S": float(nA / nS) if nS else float("nan"),
                     "frac_energy_in_S": float(nS**2 / (nS**2 + nA**2)) if (nS or nA) else float("nan")}
        # unsigned version for reference (pure structure + magnitude)
        Wu = sp.csr_matrix((w, (r, c)), shape=(n, n))
        Su = (Wu + Wu.T) * 0.5
        Au = (Wu - Wu.T) * 0.5
        out[name]["unsigned_ratio_A_over_S"] = float(sp.linalg.norm(Au, "fro") / sp.linalg.norm(Su, "fro"))
    # reciprocal pairs: same vs opposite sign (both endpoint signs known)
    key = e["body_pre"].astype(np.int64) * (1 << 32) + e["body_post"].astype(np.int64)
    rkey = e["body_post"].astype(np.int64) * (1 << 32) + e["body_pre"].astype(np.int64)
    keyset = set(key.to_numpy().tolist())
    has_rev = np.fromiter((k in keyset for k in rkey.to_numpy()), bool, len(e))
    rec = e[has_rev & (e["body_pre"] < e["body_post"])]  # one row per unordered reciprocal pair
    sign_post = nodes.set_index("bodyId")["sign"].reindex(rec["body_post"]).to_numpy()
    same = (rec["sign_pre"].to_numpy() == sign_post)
    out["reciprocal_pairs"] = {"n_pairs": int(len(rec)), "n_same_sign": int(same.sum()),
                               "n_opposite_sign": int((~same).sum()),
                               "frac_same_sign": float(same.mean()) if len(rec) else float("nan")}
    rp = rec.assign(same=same)
    rp["pair"] = ["-".join(sorted((str(a), str(b)), key=ROLES.index)) for a, b in zip(rp["role_pre"], rp["role_post"])]
    out["reciprocal_pairs"]["by_role_pair"] = rp.groupby("pair")["same"].agg(n="size", frac_same="mean")
    return out


def bm_feasibility(edges: pd.DataFrame, nodes: pd.DataFrame, dens: dict) -> dict:
    kc_by_side = nodes[nodes["role"] == "KC"].groupby("side").size()
    side = str(kc_by_side.idxmax())
    n = nodes[nodes["side"] == side].groupby("role").size().reindex(ROLES).fillna(0).astype(int)
    e = edges[(edges["side_pre"] == side) & (edges["side_post"] == side)]
    out = {"side": side, "n_PN": int(n["PN"]), "n_KC": int(n["KC"]), "n_MBON": int(n["MBON"]),
           "n_APL": int(n["APL"]), "n_DAN": int(n["DAN"])}

    def union_pairs(a, b, th):
        fwd = e[(e["role_pre"] == a) & (e["role_post"] == b) & (e["weight"] >= th)]
        bwd = e[(e["role_pre"] == b) & (e["role_post"] == a) & (e["weight"] >= th)]
        return set(map(tuple, fwd[["body_pre", "body_post"]].to_numpy())) | \
               set(map(tuple, bwd[["body_post", "body_pre"]].to_numpy()))

    for th in THRESHOLDS:
        pk = len(union_pairs("PN", "KC", th))
        km = len(union_pairs("KC", "MBON", th))
        ak = len(union_pairs("APL", "KC", th))
        kk = len(union_pairs("KC", "KC", th) | {(b, a) for a, b in union_pairs("KC", "KC", th)}) // 2
        out[f"w{th}"] = {
            "PN-KC_pairs": pk, "KC-MBON_pairs": km, "APL-KC_pairs": ak, "KC-KC_direct_pairs": kk,
            "couplings_PNKC+KCMBON": pk + km,
            "couplings_PNKC+KCMBON+APLKC": pk + km + ak,
            "couplings_PNKC+KCMBON+KCKC_direct": pk + km + kk,
            "KC_with_any_PN": int(pd.Series([b for _, b in union_pairs("PN", "KC", th)]).nunique()),
            "KC_with_any_MBON": int(pd.Series([a for a, _ in union_pairs("KC", "MBON", th)]).nunique()),
        }
    out["dense_reference"] = {
        "purchase_world_layers": PW_LAYERS,
        "purchase_world_dense_couplings": int(sum(a * b for a, b in zip(PW_LAYERS[:-1], PW_LAYERS[1:]))),
        "dense_72xKC": int(72 * n["KC"]),
        "dense_KCxMBON": int(n["KC"] * n["MBON"]),
        "dense_72xKC+KCxMBON": int(72 * n["KC"] + n["KC"] * n["MBON"]),
        "dense_PNxKC+KCxMBON": int(n["PN"] * n["KC"] + n["KC"] * n["MBON"]),
        "dense_KCxKC_over_2": int(n["KC"] * (n["KC"] - 1) // 2),
    }
    out["note_visible_mapping"] = ("Purchase World has 72 visible features; they must be mapped onto the "
                                   f"{int(n['PN'])} PN units of side {side} (72 < n_PN: choose a subset of PNs, "
                                   "or a 72 -> n_PN fixed embedding, or merge PNs by glomerulus/type).")
    # PN grouping by type (glomerulus) as a candidate mapping
    pn = nodes[(nodes["role"] == "PN") & (nodes["side"] == side)]
    out["n_PN_types_on_side"] = int(pn["type"].nunique())
    return out


def data_notes(edges: pd.DataFrame, nodes: pd.DataFrame) -> dict:
    """Automatically computed observations that need a human reading."""
    kc = nodes[nodes["role"] == "KC"]
    pk = edges[(edges["role_pre"] == "PN") & (edges["role_post"] == "KC")]
    zero = kc[~kc["bodyId"].isin(set(pk["body_post"]))]
    kc_types = kc["type"].value_counts()
    zero_types = zero["type"].value_counts()
    zt = pd.DataFrame({"n_zero_PN": zero_types, "n_type_total": kc_types.reindex(zero_types.index)})
    zt["frac_of_type"] = zt["n_zero_PN"] / zt["n_type_total"]
    dk = edges[(edges["role_pre"] == "DAN") & (edges["role_post"] == "KC")]
    dan_side = pd.crosstab(dk["side_pre"], dk["side_post"])
    dan_ipsi = float((dk["side_pre"] == dk["side_post"]).mean()) if len(dk) else float("nan")
    dpm = nodes[nodes["role"] == "DPM"][["bodyId", "consensus_nt", "predicted_nt", "predicted_nt_confidence"]]
    pn_nt_types = nodes[nodes["role"] == "PN"].groupby("nt")["type"].nunique()
    loops = edges[edges["body_pre"] == edges["body_post"]][["body_pre", "role_pre", "weight"]]
    return {"kc_zero_pn_by_type": zt, "n_kc_zero_pn": int(len(zero)), "dan_kc_side_crosstab": dan_side,
            "dan_kc_ipsilateral_frac": dan_ipsi, "dpm_nt": dpm, "pn_types_by_nt": pn_nt_types, "self_loops": loops}


# ----------------------------------------------------------------------------- report
def write_report(path: Path, st: dict, nodes: pd.DataFrame, edges: pd.DataFrame):
    L = []
    P = L.append
    P("# Step 0: MaleCNS v1.0 mushroom-body subgraph statistics")
    P("")
    P("Generated by `scripts/step0_extract_mb.py`. Text and tables only. Source: MaleCNS v1.0 flat connectome "
      "(CC-BY 4.0; files are not stored in this repository).")
    P("")
    P("## Definitions and fixed thresholds")
    P("")
    P("- Node set: `status == 'Traced'` and (`class` in {ALPN, Kenyon_Cell, MBON, DAN} or `type` in {APL, DPM}).")
    P("- Roles: PN = ALPN, KC = Kenyon_Cell, MBON, DAN, APL, DPM. Side = `somaSide` (L/R/M/NaN).")
    P("- Edge weight = synapse count of the directed pre -> post connection (min confidence 0.5, traced-only file).")
    P("- Thresholds reported: weight >= 1 (all), >= 5 (common 'significant' cutoff), >= 10.")
    P("- Sign: `consensus_nt`, falling back to `celltype_predicted_nt` when consensus is unclear. "
      "acetylcholine -> +1, gaba -> -1, glutamate -> -1 (fly glutamate mostly inhibitory), other/unclear -> 0 (unknown).")
    P("- 'Per side' tables use edges whose pre and post somas are on the same side.")
    P("")
    P("## Data provenance checks")
    P("")
    a = st["annotation"]; ei = st["edges_info"]
    P(f"- Annotation rows: {a['annotation_rows_total']:,}; Traced: {a['status_traced_total']:,}; "
      f"selected classes/types that are not Traced: {a['not_traced_in_selected_classes_or_types']:,}.")
    P(f"- `class` of APL bodies: {a['apl_dpm_class_values']['APL']}; of DPM bodies: {a['apl_dpm_class_values']['DPM']} "
      f"(superclass APL: {a['apl_dpm_superclass_values']['APL']}, DPM: {a['apl_dpm_superclass_values']['DPM']}).")
    P(f"- Edge table rows: {ei['edge_rows_total']:,} ({ei['synapses_total']:,} synapses). Induced MB edges: "
      f"{ei['induced_edges']:,} ({ei['induced_synapses']:,} synapses). Duplicate (pre,post) rows: "
      f"{ei['duplicate_pre_post_rows']}; self-loops: {ei['self_loops']}.")
    P(f"- Boundary edges (one endpoint outside the set): MB -> outside {ei['boundary_out_edges']:,} edges / "
      f"{ei['boundary_out_synapses']:,} synapses; outside -> MB {ei['boundary_in_edges']:,} edges / "
      f"{ei['boundary_in_synapses']:,} synapses.")
    P("")
    P("### Fraction of each role's synapses retained inside the subgraph")
    P("")
    P(md_table(st["boundary_by_role"], "role"))
    P("")
    P("## Node counts by role x side")
    P("")
    P(md_table(st["node_counts"], "role"))
    P("")
    P("## Edge counts and synapse totals by role pair (rows = pre, columns = post)")
    P("")
    for th in THRESHOLDS:
        P(f"### weight >= {th}: edges")
        P("")
        P(md_table(st["pair_tables"][f"w{th}"]["edges"], "pre \\ post"))
        P("")
        P(f"### weight >= {th}: synapses")
        P("")
        P(md_table(st["pair_tables"][f"w{th}"]["synapses"], "pre \\ post"))
        P("")
    P("## Core role pairs per hemisphere (same-side edges)")
    P("")
    ps = st["per_side_core"]
    for side in ["L", "R"]:
        P(f"### Side {side}")
        P("")
        P(md_table(pd.DataFrame(ps[side]).T, "pair"))
        P("")
    P(f"Cross-side edges (pre and post on different sides, incl. M/NaN): {ps['cross_side_edges']:,} edges, "
      f"{ps['cross_side_synapses']:,} synapses. By role pair:")
    P("")
    P(md_table(ps["cross_side_by_pair"], "pre \\ post"))
    P("")
    P("## KC in-degree from PN (distinct PN partners per KC)")
    P("")
    P("KCs with no PN partner at the threshold are counted with degree 0 (`n_kc_zero`); `all_nonzero` excludes them.")
    P("")
    kd = st["kc_indegree"]
    rows = {}
    for th in THRESHOLDS:
        for k in ["all", "all_nonzero", "L", "R"]:
            rows[f"w>={th} {k}"] = kd[f"w{th}"][k]
    P(md_table(pd.DataFrame(rows).T, "subset"))
    P("")
    m1 = kd["w1"]["all"]; m5 = kd["w5"]["all"]
    P(f"Literature: ~6 claws (PN inputs) per KC. Here, at weight >= 1 the mean is {m1['mean']:.2f} "
      f"(median {m1['q50']:.0f}); at weight >= 5 the mean is {m5['mean']:.2f} (median {m5['q50']:.0f}). "
      + ("The >= 5 figure agrees with the literature range." if 4 <= m5["mean"] <= 9 else
         "The >= 5 figure does not sit in the 4-9 range; see the surprises section."))
    P("")
    P("## Bipartite mask densities per hemisphere")
    P("")
    P("Density = edges / (n_pre * n_post) among same-side nodes. 'union' = undirected pair exists in either direction "
      "(what a symmetric BM mask would use).")
    P("")
    P(md_table(pd.DataFrame(st["densities"]), "metric"))
    P("")
    P("## Reciprocity")
    P("")
    r = st["reciprocity"]
    P(f"- Any weight: {r['any']['n_reciprocated']:,} of {r['any']['n_edges']:,} directed edges have a reverse edge "
      f"(fraction {r['any']['overall']:.4f}).")
    P(f"- Both directions >= 5: {r['both_ge5']['n_reciprocated']:,} of {r['both_ge5']['n_edges']:,} edges with weight >= 5 "
      f"have a reverse edge with weight >= 5 (fraction {r['both_ge5']['overall']:.4f}).")
    P("")
    P("### Fraction of role_pre -> role_post edges whose reverse edge exists (any weight)")
    P("")
    P(md_table(r["any"]["per_pair"], "pre \\ post"))
    P("")
    P("### Same, both directions >= 5")
    P("")
    P(md_table(r["both_ge5"]["per_pair"], "pre \\ post"))
    P("")
    P("## Sign coverage and neurotransmitter distribution")
    P("")
    si = st["sign_info"]
    P(f"- Nodes with no row in the NT table: {si['nodes_without_nt_row']}. Sign from consensus: {si['sign_from_consensus']:,}; "
      f"from cell-type fallback: {si['sign_from_celltype_fallback']:,}; still unclear: {si['nt_unclear_after_fallback']:,}.")
    P(f"- Fallback used, by role: {si['fallback_by_role']}.")
    P(f"- Nodes with unknown sign (0), by role: {si['sign_unknown_by_role']}.")
    P("")
    P(md_table(st["sign_coverage"], "role"))
    P("")
    P("### NT (after fallback) by role")
    P("")
    P(md_table(st["nt_by_role"], "role"))
    P("")
    P("## Symmetric / antisymmetric decomposition of the signed weighted adjacency")
    P("")
    sa = st["sym_antisym"]
    P(f"W[u,v] = sign(u) * f(weight(u->v)); nodes with unknown sign excluded ({sa['n_nodes_excluded_unknown_sign']} nodes, "
      f"{sa['n_edges_dropped_unknown_sign']:,} edges dropped; {sa['n_nodes_with_sign']:,} nodes, {sa['n_edges_used']:,} edges used). "
      "S = (W + W^T)/2, A = (W - W^T)/2.")
    P("")
    P(md_table(pd.DataFrame({k: sa[k] for k in ["raw", "log1p"]}).T, "f(weight)"))
    P("")
    rp = sa["reciprocal_pairs"]
    P(f"Among {rp['n_pairs']:,} reciprocal (unordered) pairs with both signs known: same sign {rp['n_same_sign']:,} "
      f"({rp['frac_same_sign']:.4f}), opposite sign {rp['n_opposite_sign']:,}.")
    P("")
    P(md_table(rp["by_role_pair"], "role pair"))
    P("")
    P("## BM feasibility")
    P("")
    bf = st["bm_feasibility"]
    P(f"Hemisphere with more KCs: side {bf['side']}. n_PN = {bf['n_PN']}, n_KC = {bf['n_KC']}, n_MBON = {bf['n_MBON']}, "
      f"n_APL = {bf['n_APL']}, n_DAN = {bf['n_DAN']}. Distinct PN types (glomerulus-level) on this side: {bf['n_PN_types_on_side']}.")
    P("")
    P(md_table(pd.DataFrame({f"w>={th}": bf[f"w{th}"] for th in THRESHOLDS}).T, "threshold"))
    P("")
    dr = bf["dense_reference"]
    P("Dense references:")
    P("")
    P(f"- Purchase-World DBM layers {dr['purchase_world_layers']}: {dr['purchase_world_dense_couplings']:,} couplings.")
    P(f"- Dense 72 x n_KC = {dr['dense_72xKC']:,}; dense n_KC x n_MBON = {dr['dense_KCxMBON']:,}; sum = {dr['dense_72xKC+KCxMBON']:,}.")
    P(f"- Dense n_PN x n_KC + n_KC x n_MBON = {dr['dense_PNxKC+KCxMBON']:,}; dense KC-KC (all pairs) = {dr['dense_KCxKC_over_2']:,}.")
    P("")
    P(f"- {bf['note_visible_mapping']}")
    P("")
    P("## Notes and surprises (auto-generated observations)")
    P("")
    nt_ = st["notes"]
    P(f"- {nt_['n_kc_zero_pn']} KCs receive no ALPN input at all. By KC type (they are almost entirely KCg-d and KCab-p, "
      "the visual/thermo-hygro KC subtypes whose inputs are non-olfactory PNs outside the ALPN class):")
    P("")
    P(md_table(nt_["kc_zero_pn_by_type"], "KC type"))
    P("")
    P(f"- DAN -> KC edges are only {nt_['dan_kc_ipsilateral_frac']:.3f} ipsilateral by soma side; DAN soma side is not a "
      "reliable hemisphere label for their MB innervation. Crosstab (rows = DAN soma side, cols = KC side):")
    P("")
    P(md_table(nt_["dan_kc_side_crosstab"], "DAN side"))
    P("")
    P("- DPM is predicted dopaminergic by the NT classifier (literature: serotonergic/GABAergic). Its sign is left unknown (0) here.")
    P("")
    P(md_table(nt_["dpm_nt"].set_index("bodyId"), "DPM bodyId"))
    P("")
    P(f"- PN types by NT: {nt_['pn_types_by_nt'].to_dict()} (GABAergic PNs are the multiglomerular inhibitory mlPNs; "
      "they mostly bypass KCs and target the lateral horn).")
    P(f"- Self-loops present in the file: {len(nt_['self_loops'])} (autapses; dropped from the S/A decomposition).")
    P("- The raw-weight ratio ||A||/||S|| exceeds 1 while the unsigned ratio is below 1: the dominant reciprocal loops "
      "(KC <-> APL, KC <-> MBON) pair an excitatory direction with an inhibitory return, so sign(u)*w(u->v) and "
      "sign(v)*w(v->u) have opposite signs and land almost entirely in A. See the same-sign table above: KC-APL 0, KC-MBON low.")
    P("- MBON -> KC direct edges are numerous at weight >= 1 but almost vanish at weight >= 5: not a usable feedback mask.")
    P("- KC -> KC direct edges are very dense at weight >= 1 (hundreds of thousands) but collapse at >= 5 and nearly "
      "disappear at >= 10; they are mostly weak axo-axonic contacts in the peduncle/lobes.")
    P("")
    P("## Timing and memory")
    P("")
    P(md_table(pd.DataFrame({"seconds": st["timing"]}), "step"))
    P("")
    P(f"Peak RSS: {st['peak_rss_gb']:.2f} GB.")
    P("")
    P("## Output files")
    P("")
    for k, v in st["outputs"].items():
        P(f"- {k}: `{v}`")
    P("")
    path.write_text("\n".join(L), encoding="utf-8")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR_DEFAULT)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR_DEFAULT)
    ap.add_argument("--repo-dir", type=Path, default=REPO_DIR_DEFAULT,
                    help="repository root; the report goes to <repo>/docs/step0_mb_stats.md")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.repo_dir / "docs").mkdir(parents=True, exist_ok=True)
    T = Timer()

    nodes, ann_info = load_nodes(args.data_dir)
    T.mark("load_nodes")
    nodes, sign_info = attach_signs(nodes, args.data_dir)
    T.mark("attach_signs")
    edges, bnd, edges_info = load_edges(nodes, args.data_dir)
    T.mark("load_edges")

    nodes = nodes.merge(bnd, on="bodyId", how="left")
    st: dict = {"annotation": ann_info, "sign_info": sign_info, "edges_info": edges_info}
    st["node_counts"] = nodes.groupby(["role", "side"]).size().unstack("side") \
        .reindex(index=ROLES, columns=SIDES).fillna(0).astype(int)
    st["node_counts"]["total"] = st["node_counts"].sum(axis=1)
    st["boundary_by_role"] = boundary_by_role(nodes, bnd)
    st["pair_tables"] = {}
    for th in THRESHOLDS:
        e = edges[edges["weight"] >= th]
        st["pair_tables"][f"w{th}"] = {"edges": role_pair_table(e, None), "synapses": role_pair_table(e, "weight")}
    T.mark("pair_tables")
    st["per_side_core"] = per_side_core(edges, nodes)
    st["kc_indegree"] = kc_indegree(edges, nodes)
    st["densities"] = densities(edges, nodes)
    T.mark("side_degree_density")
    st["reciprocity"] = reciprocity(edges)
    T.mark("reciprocity")
    cov = nodes.groupby("role").agg(n=("sign", "size"), n_sign_known=("sign", lambda s: int((s != 0).sum())),
                                    n_excit=("sign", lambda s: int((s > 0).sum())),
                                    n_inhib=("sign", lambda s: int((s < 0).sum())))
    cov["frac_sign_known"] = cov["n_sign_known"] / cov["n"]
    st["sign_coverage"] = cov.reindex(ROLES)
    st["nt_by_role"] = nodes.groupby(["role", "nt"]).size().unstack("nt").reindex(ROLES).fillna(0).astype(int)
    st["sym_antisym"] = sym_antisym(edges, nodes)
    T.mark("sym_antisym")
    st["bm_feasibility"] = bm_feasibility(edges, nodes, st["densities"])
    st["notes"] = data_notes(edges, nodes)
    T.mark("bm_feasibility")

    # save data products
    node_cols = ["bodyId", "role", "side", "somaSide", "type", "class", "superclass", "instance", "group",
                 "nt", "nt_source", "sign", "consensus_nt", "celltype_predicted_nt", "predicted_nt",
                 "predicted_nt_confidence", "int_in_edges", "int_in_syn", "int_out_edges", "int_out_syn",
                 "ext_in_edges", "ext_in_syn", "ext_out_edges", "ext_out_syn"]
    p_nodes = args.out_dir / "mb_nodes.parquet"
    p_edges = args.out_dir / "mb_edges.parquet"
    p_json = args.out_dir / "mb_stats.json"
    p_md = args.repo_dir / "docs" / "step0_mb_stats.md"
    nodes[node_cols].to_parquet(p_nodes, index=False)
    edges.assign(role_pre=edges["role_pre"].astype(str), role_post=edges["role_post"].astype(str)) \
         .to_parquet(p_edges, index=False)
    T.mark("save_parquet")
    st["timing"] = dict(T.marks)
    st["peak_rss_gb"] = peak_rss_gb()
    st["outputs"] = {"nodes": str(p_nodes), "edges": str(p_edges), "stats_json": str(p_json), "report": str(p_md)}
    p_json.write_text(json.dumps(to_jsonable(st), indent=1), encoding="utf-8")
    write_report(p_md, st, nodes, edges)
    T.mark("write_report")

    # stdout summary
    nc = st["node_counts"]
    d = st["densities"]; bf = st["bm_feasibility"]; sa = st["sym_antisym"]; r = st["reciprocity"]
    print("\n=== Step 0 summary ===")
    print(nc.to_string())
    print(f"induced edges {edges_info['induced_edges']:,}, synapses {edges_info['induced_synapses']:,}")
    for s in ["L", "R"]:
        print(f"side {s}: PN->KC edges w>=1 {d[s]['PN->KC_edges_w1']:,} (density {d[s]['PN->KC_density_w1']:.4f}), "
              f"w>=5 {d[s]['PN->KC_edges_w5']:,} ({d[s]['PN->KC_density_w5']:.4f}); "
              f"KC->MBON w>=1 {d[s]['KC->MBON_edges_w1']:,} ({d[s]['KC->MBON_density_w1']:.4f}), "
              f"w>=5 {d[s]['KC->MBON_edges_w5']:,} ({d[s]['KC->MBON_density_w5']:.4f})")
    k1 = st["kc_indegree"]["w1"]["all"]; k5 = st["kc_indegree"]["w5"]["all"]
    print(f"KC in-degree from PN: w>=1 mean {k1['mean']:.2f} median {k1['q50']:.0f}; w>=5 mean {k5['mean']:.2f} median {k5['q50']:.0f}")
    print(f"reciprocity any {r['any']['overall']:.4f}, both>=5 {r['both_ge5']['overall']:.4f}")
    print(f"sign known: {st['sign_coverage']['frac_sign_known'].round(3).to_dict()}")
    print(f"|A|/|S| raw {sa['raw']['ratio_A_over_S']:.4f}, log1p {sa['log1p']['ratio_A_over_S']:.4f}; "
          f"reciprocal same-sign frac {sa['reciprocal_pairs']['frac_same_sign']:.4f}")
    print(f"BM side {bf['side']}: PN {bf['n_PN']}, KC {bf['n_KC']}, MBON {bf['n_MBON']}; "
          f"couplings PN-KC+KC-MBON w>=1 {bf['w1']['couplings_PNKC+KCMBON']:,}, w>=5 {bf['w5']['couplings_PNKC+KCMBON']:,}")
    print(f"peak RSS {st['peak_rss_gb']:.2f} GB; report: {p_md}")


if __name__ == "__main__":
    main()
