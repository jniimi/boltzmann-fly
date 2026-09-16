"""Masked Bernoulli-Bernoulli RBM / DBM on top of the vendored Purchase World classes.

Masking contract (enforced three times over):
  1. every forward use of a weight matrix goes through `effective_weight()`, which multiplies
     the raw parameter by a fixed 0/1 buffer (and, with `dale=True`, by a fixed sign matrix
     applied to |W|);
  2. a gradient hook multiplies the incoming gradient by the mask, so masked entries never
     receive a non-zero gradient (Adam moments stay exactly zero);
  3. after every `train_step` (the only place the vendored code calls `optimizer.step()`),
     and after `initialize_from_rbms` / `load_state_dict`, the raw parameter is re-masked in
     place, so `state_dict()` also contains exact zeros at masked entries.

Only the Bernoulli-Bernoulli, non-recurrent code paths of the upstream classes are
supported (`use_gbrbm=False`, `lag_window=0`); these are the paths used by the ICONIP run.

`masked_classes(...)` is a context manager that swaps the class names looked up inside the
vendored `greedy_pretrain_dbm` so the *training loops themselves stay byte-identical* to
upstream while constructing masked models.
"""
from __future__ import annotations

import contextlib
from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .vendor import dbm as _vdbm
from .vendor.dbm import DeepBoltzmannMachine
from .vendor.rbm import RestrictedBoltzmannMachine


def _as_mask(mask, shape, device=None) -> torch.Tensor:
    if mask is None:
        m = torch.ones(shape)
    else:
        m = torch.as_tensor(np.asarray(mask), dtype=torch.float32)
        assert tuple(m.shape) == tuple(shape), (tuple(m.shape), tuple(shape))
        m = (m != 0).float()
    return m.to(device) if device is not None else m


@torch.no_grad()
def fanin_init_(W: torch.Tensor, mask: torch.Tensor, sigma: float = 1.0, generator=None):
    """W_ij ~ N(0, sigma^2 / fan_in_j), fan_in_j = number of unmasked inputs of hidden unit j.

    With binary inputs the initial drive of unit j, sum_i v_i W_ij, then has variance
    sigma^2 * (#active inputs / fan_in_j) <= sigma^2, i.e. O(1) regardless of the fan-in
    (upstream uses N(0, 0.01^2) for every entry, which gives a drive of ~0.01*sqrt(fan_in)).
    """
    fan_in = mask.sum(0).clamp(min=1.0)  # (n_hidden,)
    W.normal_(0.0, 1.0, generator=generator)
    W.mul_(sigma / fan_in.sqrt()[None, :]).mul_(mask)


def _sign_matrix(pre_sign, shape) -> torch.Tensor:
    """Fixed sign matrix from the presynaptic (row) unit's sign; unknown sign (0) -> +1."""
    s = torch.as_tensor(np.asarray(pre_sign), dtype=torch.float32)
    assert s.shape == (shape[0],)
    s = torch.where(s == 0, torch.ones_like(s), torch.sign(s))
    return s[:, None].expand(*shape).contiguous()


class _MaskGradHook:
    """Picklable gradient hook: g -> g * module.<mask_attr>."""

    def __init__(self, module: nn.Module, mask_attr: str):
        self.module = module
        self.mask_attr = mask_attr

    def __call__(self, g):
        return g * getattr(self.module, self.mask_attr)


def _has_mask_hook(param: torch.Tensor) -> bool:
    hooks = getattr(param, "_backward_hooks", None) or {}
    return any(isinstance(h, _MaskGradHook) for h in hooks.values())


class MaskedRBM(RestrictedBoltzmannMachine):
    """Bernoulli-Bernoulli RBM whose coupling W is restricted to a fixed 0/1 pattern."""

    def __init__(self, n_visible, n_hidden, mask=None, dale: bool = False, pre_sign=None,
                 init: str = "default", init_sigma: float = 1.0):
        super().__init__(n_visible, n_hidden)
        self.register_buffer("mask", _as_mask(mask, (n_visible, n_hidden)))
        self.dale = bool(dale)
        self.init = init
        if self.dale:
            self.register_buffer("sign", _sign_matrix(pre_sign, (n_visible, n_hidden)))
        if init == "fanin":
            fanin_init_(self.W.data, self.mask, init_sigma)
        elif init != "default":
            raise ValueError(init)
        self.apply_mask_()
        self._register_hooks()

    # ---- masking machinery
    def _register_hooks(self):
        if not _has_mask_hook(self.W):
            self.W.register_hook(_MaskGradHook(self, "mask"))

    def __setstate__(self, state):
        super().__setstate__(state)
        self._register_hooks()
        self.apply_mask_()

    @torch.no_grad()
    def apply_mask_(self):
        self.W.data.mul_(self.mask)

    def effective_weight(self) -> torch.Tensor:
        W = self.W
        if self.dale:
            W = W.abs() * self.sign
        return W * self.mask

    def load_state_dict(self, *a, **k):
        r = super().load_state_dict(*a, **k)
        self.apply_mask_()
        return r

    # ---- forward uses (mirror upstream rbm.py exactly, with W -> effective_weight())
    def sample_h_given_v(self, v):
        activation = torch.matmul(v, self.effective_weight()) + self.h_bias
        prob = torch.sigmoid(activation)
        return torch.bernoulli(prob), prob

    def sample_v_given_h(self, h):
        activation = torch.matmul(h, self.effective_weight().t()) + self.v_bias
        prob = torch.sigmoid(activation)
        return torch.bernoulli(prob), prob

    def free_energy(self, v):
        v_term = torch.matmul(v, self.v_bias)
        wx_b = torch.matmul(v, self.effective_weight()) + self.h_bias
        hidden_term = torch.sum(F.softplus(wx_b), dim=1)
        return -v_term - hidden_term

    def train_step(self, v_pos, optimizer, pcd_chain, k_steps=1):
        out = super().train_step(v_pos, optimizer, pcd_chain, k_steps)
        self.apply_mask_()
        return out


class MaskedDBM(DeepBoltzmannMachine):
    """Bernoulli-Bernoulli DBM with one fixed 0/1 mask per coupling layer."""

    def __init__(self, layer_sizes, masks: Optional[Sequence] = None, dale: bool = False,
                 pre_signs: Optional[Sequence] = None, init: str = "default", init_sigma: float = 1.0, **kwargs):
        assert not kwargs.get("use_gbrbm", False) and kwargs.get("lag_window", 0) == 0, \
            "MaskedDBM supports only the Bernoulli-Bernoulli, non-recurrent DBM"
        super().__init__(layer_sizes, **kwargs)
        self.dale = bool(dale)
        self.init = init
        for i in range(self.n_layers):
            shape = (layer_sizes[i], layer_sizes[i + 1])
            m = None if masks is None else masks[i]
            self.register_buffer(f"mask_{i}", _as_mask(m, shape))
            if self.dale:
                s = None if pre_signs is None else pre_signs[i]
                if s is None:
                    s = np.ones(shape[0])
                self.register_buffer(f"sign_{i}", _sign_matrix(s, shape))
            if init == "fanin":  # overwritten by initialize_from_rbms in the normal pipeline
                fanin_init_(self.weights[i].data, getattr(self, f"mask_{i}"), init_sigma)
        self.apply_mask_()
        self._register_hooks()

    # ---- masking machinery
    def _register_hooks(self):
        for i in range(self.n_layers):
            if not _has_mask_hook(self.weights[i]):
                self.weights[i].register_hook(_MaskGradHook(self, f"mask_{i}"))

    def __setstate__(self, state):
        super().__setstate__(state)
        self._register_hooks()
        self.apply_mask_()

    def mask(self, i: int) -> torch.Tensor:
        return getattr(self, f"mask_{i}")

    @torch.no_grad()
    def apply_mask_(self):
        for i in range(self.n_layers):
            self.weights[i].data.mul_(self.mask(i))

    def effective_weight(self, i: int) -> torch.Tensor:
        W = self.weights[i]
        if self.dale:
            W = W.abs() * getattr(self, f"sign_{i}")
        return W * self.mask(i)

    def effective_weights(self) -> List[torch.Tensor]:
        return [self.effective_weight(i) for i in range(self.n_layers)]

    def initialize_from_rbms(self, rbm_list, weight_scaling=None):
        # copy raw parameters from the (already masked) RBMs, then re-mask (belt and braces)
        super().initialize_from_rbms(rbm_list, weight_scaling=weight_scaling)
        self.apply_mask_()

    def load_state_dict(self, *a, **k):
        r = super().load_state_dict(*a, **k)
        self.apply_mask_()
        return r

    def drive_stats(self) -> dict:
        """Per hidden unit: sum of |w_eff| over its fan-in (how much the layer below can move it)."""
        out = {}
        for i in range(self.n_layers):
            W = self.effective_weight(i).detach()
            drive = W.abs().sum(0)
            out[f"drive_median_l{i + 1}"] = float(drive.median())
            out[f"drive_mean_l{i + 1}"] = float(drive.mean())
            out[f"fanin_median_l{i + 1}"] = float((W != 0).float().sum(0).median())
            nz = W[W != 0]
            out[f"abs_w_mean_l{i + 1}"] = float(nz.abs().mean()) if nz.numel() else 0.0
        return out

    def n_couplings(self) -> int:
        return int(sum(int(self.mask(i).sum().item()) for i in range(self.n_layers)))

    def n_params_effective(self) -> int:
        return self.n_couplings() + int(sum(b.numel() for b in self.biases))

    # ---- forward uses (mirror upstream dbm.py, Bernoulli path, with weights[i] -> effective_weight(i))
    def compensate_biases(self, data, device, batch_size=256):
        if self.n_layers <= 1:
            print("   (Bias compensation skipped: single hidden layer, no top-down)")
            return
        self.eval()
        W = self.effective_weights()
        all_h_means = [[] for _ in range(self.n_layers)]
        with torch.no_grad():
            for i in range(0, len(data), batch_size):
                batch = data[i:i + batch_size].to(device)
                activation = torch.matmul(batch, W[0]) + self.biases[1]
                h_prob = torch.sigmoid(activation)
                all_h_means[0].append(h_prob.cpu())
                for layer_idx in range(1, self.n_layers):
                    activation = torch.matmul(h_prob, W[layer_idx]) + self.biases[layer_idx + 1]
                    h_prob = torch.sigmoid(activation)
                    all_h_means[layer_idx].append(h_prob.cpu())
        expected_h = [torch.cat(all_h_means[l], dim=0).mean(dim=0) for l in range(self.n_layers)]
        print(f"   Bias compensation:")
        for layer_idx in range(self.n_layers - 1):
            e_h_above = expected_h[layer_idx + 1].to(device)
            correction = torch.matmul(e_h_above, W[layer_idx + 1].t())
            before_mean = self.biases[layer_idx + 1].data.mean().item()
            self.biases[layer_idx + 1].data -= correction
            after_mean = self.biases[layer_idx + 1].data.mean().item()
            print(f"     L{layer_idx+1}: E[h{layer_idx+2}]={e_h_above.mean():.4f}, "
                  f"correction norm={correction.norm():.4f}, "
                  f"bias mean {before_mean:.4f} → {after_mean:.4f}")

    def mean_field_inference(self, v, n_iter=10):
        batch_size = v.shape[0]
        W = self.effective_weights()
        h_probs = []
        for i in range(self.n_layers):
            h_prob = torch.ones(batch_size, self.layer_sizes[i + 1], device=v.device) * self.mf_init[i]
            h_probs.append(h_prob)
        v_scaled = v
        for _ in range(n_iter):
            bottom_up = torch.matmul(v_scaled, W[0]) + self.biases[1]
            if self.n_layers > 1:
                top_down = torch.matmul(h_probs[1], W[1].t())
                h_probs[0] = torch.sigmoid(bottom_up + top_down)
            else:
                h_probs[0] = torch.sigmoid(bottom_up)
            for i in range(1, self.n_layers - 1):
                bottom_up = torch.matmul(h_probs[i - 1], W[i]) + self.biases[i + 1]
                top_down = torch.matmul(h_probs[i + 1], W[i + 1].t())
                h_probs[i] = torch.sigmoid(bottom_up + top_down)
            if self.n_layers > 1:
                bottom_up = torch.matmul(h_probs[-2], W[-1]) + self.biases[-1]
                h_probs[-1] = torch.sigmoid(bottom_up)
        return h_probs

    def sample_v_given_h(self, h_list, add_noise=True):
        activation = torch.matmul(h_list[0], self.effective_weight(0).t()) + self.biases[0]
        v_prob = torch.sigmoid(activation)
        v_sample = torch.bernoulli(v_prob)
        return v_sample, v_prob

    def free_energy(self, v, n_iter=10):
        with torch.no_grad():
            h_probs_detached = self.mean_field_inference(v, n_iter)
        h_probs = [h.detach() for h in h_probs_detached]
        W = self.effective_weights()
        v_term = torch.matmul(v, self.biases[0])
        v_scaled = v
        hidden_term = 0.0
        wx_b = torch.matmul(v_scaled, W[0]) + self.biases[1]
        if self.n_layers > 1:
            wx_b += torch.matmul(h_probs[1], W[1].t())
        hidden_term += torch.sum(h_probs[0] * wx_b, dim=1)
        for i in range(1, self.n_layers - 1):
            wx_b = torch.matmul(h_probs[i - 1], W[i]) + self.biases[i + 1]
            wx_b += torch.matmul(h_probs[i + 1], W[i + 1].t())
            hidden_term += torch.sum(h_probs[i] * wx_b, dim=1)
        if self.n_layers > 1:
            wx_b = torch.matmul(h_probs[-2], W[-1]) + self.biases[-1]
            hidden_term += torch.sum(h_probs[-1] * wx_b, dim=1)
        return -v_term - hidden_term

    def train_step(self, v_pos, optimizer, pcd_buffer, k_steps=5, n_iter=10, grad_clip=None):
        out = super().train_step(v_pos, optimizer, pcd_buffer, k_steps=k_steps, n_iter=n_iter, grad_clip=grad_clip)
        self.apply_mask_()
        return out


# ----------------------------------------------------------------------------- pipeline hook
@contextlib.contextmanager
def masked_classes(masks: Optional[Sequence], dale: bool = False, pre_signs: Optional[Sequence] = None,
                   init: str = "default", init_sigma: float = 1.0):
    """Route the vendored greedy_pretrain_dbm through MaskedRBM / MaskedDBM.

    The vendored function builds `RestrictedBoltzmannMachine(n_v, n_h)` per layer and one
    `DeepBoltzmannMachine(layer_sizes, ...)` at the end, looking the names up in its module
    globals. Inside this context those names resolve to factories that attach the mask whose
    shape matches (n_v, n_h). `masks=None` gives all-ones masks (dense, for verification).
    """
    masks = None if masks is None else [np.asarray(m) for m in masks]
    signs = None if pre_signs is None else [np.asarray(s) for s in pre_signs]

    def _find(shape):
        if masks is None:
            return None, None
        for i, m in enumerate(masks):
            if tuple(m.shape) == tuple(shape):
                return m, (None if signs is None else signs[i])
        raise ValueError(f"no mask with shape {shape}; available {[m.shape for m in masks]}")

    def rbm_factory(n_visible, n_hidden):
        m, s = _find((n_visible, n_hidden))
        return MaskedRBM(n_visible, n_hidden, mask=m, dale=dale, pre_sign=s, init=init, init_sigma=init_sigma)

    def dbm_factory(layer_sizes, **kwargs):
        pairs = [_find((layer_sizes[i], layer_sizes[i + 1])) for i in range(len(layer_sizes) - 1)]
        ms = None if masks is None else [p[0] for p in pairs]
        ss = None if signs is None else [p[1] for p in pairs]
        return MaskedDBM(layer_sizes, masks=ms, dale=dale, pre_signs=ss, init=init, init_sigma=init_sigma, **kwargs)

    saved = (_vdbm.RestrictedBoltzmannMachine, _vdbm.DeepBoltzmannMachine)
    _vdbm.RestrictedBoltzmannMachine = rbm_factory
    _vdbm.DeepBoltzmannMachine = dbm_factory
    try:
        yield
    finally:
        _vdbm.RestrictedBoltzmannMachine, _vdbm.DeepBoltzmannMachine = saved
