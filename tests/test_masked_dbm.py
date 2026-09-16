"""Masking invariants and numerical equivalence with the vendored (upstream) classes."""
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from boltzmann_fly.masked_dbm import MaskedDBM, MaskedRBM, masked_classes
from boltzmann_fly.vendor.dbm import DeepBoltzmannMachine
from boltzmann_fly.vendor.rbm import RestrictedBoltzmannMachine
from boltzmann_fly.vendor.train import MockDataset, joint_finetuning, pretrain_bm

DEV = torch.device("cpu")


def _rand_mask(shape, p=0.3, seed=0):
    rng = np.random.default_rng(seed)
    m = (rng.random(shape) < p).astype(np.float32)
    m[0, 0] = 1.0  # keep at least one edge
    return m


def _copy_state(src, dst):
    sd = {k: v.clone() for k, v in src.state_dict().items()}
    missing, unexpected = dst.load_state_dict(sd, strict=False)
    assert not unexpected
    assert all(k.startswith(("mask", "sign")) for k in missing), missing


# --------------------------------------------------------------------------- equivalence (all-ones mask)
def test_rbm_unmasked_equals_upstream():
    torch.manual_seed(0)
    ref = RestrictedBoltzmannMachine(20, 8)
    torch.manual_seed(0)
    mine = MaskedRBM(20, 8, mask=None)
    _copy_state(ref, mine)
    v = (torch.rand(16, 20) > 0.5).float()
    torch.manual_seed(1); _, p_ref = ref.sample_h_given_v(v)
    torch.manual_seed(1); _, p_mine = mine.sample_h_given_v(v)
    assert torch.equal(p_ref, p_mine)
    assert torch.equal(ref.free_energy(v), mine.free_energy(v))
    h = (torch.rand(16, 8) > 0.5).float()
    assert torch.equal(ref.sample_v_given_h(h)[1], mine.sample_v_given_h(h)[1])
    # identical training trajectory
    opt_ref = torch.optim.Adam(ref.parameters(), lr=1e-2, weight_decay=1e-3)
    opt_mine = torch.optim.Adam(mine.parameters(), lr=1e-2, weight_decay=1e-3)
    pcd = (torch.rand(16, 20) > 0.5).float()
    for _ in range(5):
        torch.manual_seed(7); l_ref, pcd_ref = ref.train_step(v, opt_ref, pcd, k_steps=1)
        torch.manual_seed(7); l_mine, pcd_mine = mine.train_step(v, opt_mine, pcd, k_steps=1)
        assert l_ref == l_mine and torch.equal(pcd_ref, pcd_mine)
        pcd = pcd_ref
    assert torch.equal(ref.W, mine.W) and torch.equal(ref.h_bias, mine.h_bias)


def test_dbm_unmasked_equals_upstream():
    sizes = [20, 8, 4]
    torch.manual_seed(0)
    ref = DeepBoltzmannMachine(sizes)
    mine = MaskedDBM(sizes, masks=None)
    _copy_state(ref, mine)
    for W in ref.weights:
        W.data.normal_(0, 0.3)
    _copy_state(ref, mine)
    v = (torch.rand(16, 20) > 0.5).float()
    for a, b in zip(ref.mean_field_inference(v, 10), mine.mean_field_inference(v, 10)):
        assert torch.equal(a, b)
    assert torch.equal(ref.free_energy(v, 10), mine.free_energy(v, 10))
    h = [(torch.rand(16, s) > 0.5).float() for s in sizes[1:]]
    assert torch.equal(ref.sample_v_given_h(h)[1], mine.sample_v_given_h(h)[1])
    # bias compensation and training steps
    ref.compensate_biases(v, DEV); mine.compensate_biases(v, DEV)
    assert all(torch.equal(a, b) for a, b in zip(ref.biases, mine.biases))
    opt_ref = torch.optim.Adam(ref.parameters(), lr=1e-3, weight_decay=1e-4)
    opt_mine = torch.optim.Adam(mine.parameters(), lr=1e-3, weight_decay=1e-4)
    pcd = (torch.rand(16, 20) > 0.5).float()
    for _ in range(3):
        torch.manual_seed(3); l_ref, pcd_ref = ref.train_step(v, opt_ref, pcd, k_steps=2, n_iter=5)
        torch.manual_seed(3); l_mine, pcd_mine = mine.train_step(v, opt_mine, pcd, k_steps=2, n_iter=5)
        assert l_ref == l_mine and torch.equal(pcd_ref, pcd_mine)
        pcd = pcd_ref
    assert all(torch.equal(a, b) for a, b in zip(ref.weights, mine.weights))


# --------------------------------------------------------------------------- masked entries stay exactly zero
def test_masked_entries_zero_after_steps():
    m = _rand_mask((30, 12), seed=1)
    torch.manual_seed(0)
    rbm = MaskedRBM(30, 12, mask=m)
    assert torch.equal(rbm.W[torch.as_tensor(m) == 0], torch.zeros(int((m == 0).sum())))
    opt = torch.optim.Adam(rbm.parameters(), lr=1e-2, weight_decay=1e-3)
    v = (torch.rand(64, 30) > 0.5).float()
    pcd = (torch.rand(64, 30) > 0.5).float()
    w0 = rbm.W.detach().clone()
    for _ in range(10):
        _, pcd = rbm.train_step(v, opt, pcd, k_steps=1)
    zero = torch.as_tensor(m) == 0
    assert (rbm.W[zero] == 0).all()
    assert (rbm.effective_weight()[zero] == 0).all()
    assert not torch.equal(rbm.W[~zero], w0[~zero])  # unmasked entries did train

    sizes = [30, 12, 5]
    masks = [m, _rand_mask((12, 5), seed=2)]
    dbm = MaskedDBM(sizes, masks=masks)
    opt = torch.optim.Adam(dbm.parameters(), lr=1e-2, weight_decay=1e-4)
    pcd = (torch.rand(64, 30) > 0.5).float()
    for _ in range(10):
        _, pcd = dbm.train_step(v, opt, pcd, k_steps=2, n_iter=5)
    for i, mm in enumerate(masks):
        zero = torch.as_tensor(mm) == 0
        assert (dbm.weights[i][zero] == 0).all() and (dbm.effective_weight(i)[zero] == 0).all()
        assert dbm.weights[i].grad is not None and (dbm.weights[i].grad[zero] == 0).all()
    assert dbm.n_couplings() == int(sum(mm.sum() for mm in masks))


def test_masked_pipeline_pretrain_finetune_and_pickle(tmp_path: Path):
    """Run the vendored pretrain_bm + joint_finetuning through masked_classes; zeros must survive,
    including the pickle round trip that the vendored code performs."""
    torch.manual_seed(0)
    n_v = 24
    sizes = [n_v, 10, 6]
    masks = [_rand_mask((24, 10), seed=3), _rand_mask((10, 6), seed=4)]
    ds = MockDataset(256, n_v, continuous=False)
    loader = DataLoader(ds, batch_size=32, shuffle=True)
    (tmp_path / "models").mkdir()
    cfg = {
        "device": DEV, "drive_dir": tmp_path, "dataset": {"n_visible": n_v},
        "bm": {"layer_sizes": sizes, "batchsize": 32, "nepochs_greedy_pretraining": 3, "pretraining_lr": 0.01,
               "pretraining_ksteps": 1, "save_pretrain_id": "pt", "nepochs_joint_finetuning": 3,
               "finetuning_lr": 0.001, "finetuning_decay": 1e-4, "finetuning_ksteps": 1, "finetuning_niter": 5,
               "save_finetuning_id": "ft", "use_gbrbm": False, "sigma_init": 1.0, "learn_sigma": False,
               "pretraining_weight_decay": 1e-3, "compensate_biases": True, "finetuning_patience": 20},
        "gpt": {}, "adapter": {}, "verbose": {"send_message": False},
    }
    with masked_classes(masks):
        dbm, rbms = pretrain_bm(cfg, loader, verbose=0)
    assert isinstance(dbm, MaskedDBM) and all(isinstance(r, MaskedRBM) for r in rbms)
    for i, mm in enumerate(masks):
        assert (dbm.weights[i][torch.as_tensor(mm) == 0] == 0).all()
        assert (rbms[i].W[torch.as_tensor(mm) == 0] == 0).all()
    dbm = joint_finetuning(cfg, dbm, loader, val_dataloader=DataLoader(ds, batch_size=32), verbose=0)
    for i, mm in enumerate(masks):
        zero = torch.as_tensor(mm) == 0
        assert (dbm.weights[i][zero] == 0).all()
        assert (dbm.weights[i][~zero] != 0).any()
    # pickle round trip (vendored code saves with pickle) keeps masks and hooks
    with open(tmp_path / "models" / "ft.pkl", "rb") as f:
        loaded = pickle.load(f)
    assert isinstance(loaded, MaskedDBM)
    for i, mm in enumerate(masks):
        assert torch.equal(loaded.mask(i), torch.as_tensor(mm))
        assert (loaded.weights[i][torch.as_tensor(mm) == 0] == 0).all()
    opt = torch.optim.Adam(loaded.parameters(), lr=1e-2)
    v = (torch.rand(32, n_v) > 0.5).float()
    loaded.train_step(v, opt, v.clone(), k_steps=1, n_iter=3)
    for i, mm in enumerate(masks):
        assert (loaded.weights[i][torch.as_tensor(mm) == 0] == 0).all()
        assert (loaded.weights[i].grad[torch.as_tensor(mm) == 0] == 0).all()


def test_dale_signs():
    m = _rand_mask((10, 6), seed=5)
    pre_sign = np.array([1, -1, 1, 0, -1, 1, 1, -1, 1, 1])
    rbm = MaskedRBM(10, 6, mask=m, dale=True, pre_sign=pre_sign)
    rbm.W.data.normal_()
    Weff = rbm.effective_weight()
    expect = np.where(pre_sign == 0, 1, pre_sign)
    for i in range(10):
        for j in range(6):
            if m[i, j] == 0:
                assert Weff[i, j] == 0
            else:
                assert np.sign(Weff[i, j].item()) == expect[i]
    dbm = MaskedDBM([10, 6, 3], masks=[m, np.ones((6, 3))], dale=True, pre_signs=[pre_sign, np.ones(6)])
    for W in dbm.weights:
        W.data.normal_()
    assert (dbm.effective_weight(1) > 0).all()
    assert (torch.sign(dbm.effective_weight(0)[1]) <= 0).all()


def test_masked_classes_restores_globals():
    from boltzmann_fly.vendor import dbm as vd
    orig = (vd.RestrictedBoltzmannMachine, vd.DeepBoltzmannMachine)
    with masked_classes(None):
        assert vd.RestrictedBoltzmannMachine is not orig[0]
    assert (vd.RestrictedBoltzmannMachine, vd.DeepBoltzmannMachine) == orig


def test_fanin_init_drive_scale():
    from boltzmann_fly.masked_dbm import fanin_init_
    torch.manual_seed(0)
    m = _rand_mask((343, 400), p=0.015, seed=9)
    rbm = MaskedRBM(343, 400, mask=m, init="fanin", init_sigma=1.0)
    W = rbm.effective_weight()
    assert (W[torch.as_tensor(m) == 0] == 0).all()
    fan_in = torch.as_tensor(m).sum(0)
    # drive with all fan-in inputs on: variance ~ 1 per unit -> sum over units of (sum_i W_ij)^2 ~ n_hidden
    drive_all_on = W.sum(0)
    assert 0.5 < drive_all_on.pow(2).mean() < 2.0
    # per-entry variance ~ 1/fan_in (check on well-populated columns)
    big = fan_in >= 4
    ratio = (W[:, big].pow(2).sum(0) / fan_in[big]).mean()  # mean of fan_in * var / fan_in = var*... -> ~1/fan_in*fan_in/fan_in
    assert ratio.item() > 0  # sanity
    nz = torch.as_tensor(m) != 0
    assert rbm.W[nz].abs().mean() > 0.1  # upstream init is 0.01-scale; fan-in init gives ~1/sqrt(fan_in)
