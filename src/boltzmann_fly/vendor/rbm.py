import os, pickle, random, gc
from pathlib import Path
from tqdm.auto import tqdm
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import *

class RestrictedBoltzmannMachine(nn.Module):
    def __init__(self, n_visible, n_hidden):
        super().__init__()
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        
        # W: (n_visible, n_hidden)
        self.W = nn.Parameter(torch.randn(n_visible, n_hidden) * 0.01)
        
        # バイアス
        self.v_bias = nn.Parameter(torch.zeros(n_visible))
        self.h_bias = nn.Parameter(torch.zeros(n_hidden))

    def sample_h_given_v(self, v):
        """P(h|v)"""
        activation = torch.matmul(v, self.W) + self.h_bias
        prob = torch.sigmoid(activation)
        return torch.bernoulli(prob), prob

    def sample_v_given_h(self, h):
        """P(v|h)"""
        activation = torch.matmul(h, self.W.t()) + self.v_bias
        prob = torch.sigmoid(activation)
        return torch.bernoulli(prob), prob

    def free_energy(self, v):
        """自由エネルギー F(v)"""
        v_term = torch.matmul(v, self.v_bias)
        wx_b = torch.matmul(v, self.W) + self.h_bias
        # Softplus = log(1 + exp(x))
        hidden_term = torch.sum(F.softplus(wx_b), dim=1)
        return -v_term - hidden_term

    def train_step(self, v_pos, optimizer, pcd_chain, k_steps=1):
        """PCD法による学習"""
        # 1. Negative Phase (Dreaming)
        # バッファ(v)からスタートして k回 往復する
        v_neg = pcd_chain.detach()
        
        for _ in range(k_steps):
            # v -> h
            h_neg, _ = self.sample_h_given_v(v_neg)
            # h -> v
            v_neg, _ = self.sample_v_given_h(h_neg)
            
        # 2. Loss Calculation (Free Energy Difference)
        # Positive(現実)の自由エネルギーを下げ、Negative(夢)を上げる
        energy_pos = self.free_energy(v_pos).mean()
        energy_neg = self.free_energy(v_neg).mean()
        
        loss = energy_pos - energy_neg
        
        # 3. Update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 更新された夢(v)を返す
        return loss.item(), v_neg
    
    # --- Adapterへの入力用 ---
    def get_latent_representation(self, v):
        _, h_prob = self.sample_h_given_v(v)
        return h_prob # 0/1ではなく確率値を渡す（情報量が多い）


class GaussianBernoulliRBM(nn.Module):
    """
    Gaussian-Bernoulli Restricted Boltzmann Machine (GBRBM)

    可視層が連続値（ガウス分布）、隠れ層が二値（ベルヌーイ分布）のRBM。
    連続値の入力データ（価格、頻度、beliefベクトル等）を扱うために使用。

    Energy function:
        E(v, h) = Σ_i (v_i - b_i)^2 / (2σ_i^2) - Σ_j c_j h_j - Σ_{i,j} (v_i / σ_i^2) W_{ij} h_j

    where:
        v: 可視層（連続値）
        h: 隠れ層（二値）
        b: 可視層バイアス
        c: 隠れ層バイアス
        W: 重み行列
        σ: 可視層の標準偏差（学習可能 or 固定）

    References:
        - Hinton, G. E. (2012). A practical guide to training restricted Boltzmann machines.
        - Cho, K., et al. (2011). Improved learning of Gaussian-Bernoulli restricted Boltzmann machines.
    """

    def __init__(self, n_visible, n_hidden, learn_sigma=False, sigma_init=1.0,
                 weight_init="xavier_normal"):
        """
        Args:
            n_visible: 可視層のユニット数
            n_hidden: 隠れ層のユニット数
            learn_sigma: Trueの場合、σを学習可能パラメータにする
            sigma_init: σの初期値
            weight_init: 重み初期化の方法
                - "xavier_normal": Xavier正規分布（デフォルト）
                - "small_random": 0.01 * randn（従来の方法）
        """
        super().__init__()
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.learn_sigma = learn_sigma

        # W: (n_visible, n_hidden)
        if weight_init == "xavier_normal":
            self.W = nn.Parameter(torch.empty(n_visible, n_hidden))
            nn.init.xavier_normal_(self.W.data)
        elif weight_init == "small_random":
            self.W = nn.Parameter(torch.randn(n_visible, n_hidden) * 0.01)
        else:
            raise ValueError(f"Unknown weight_init: {weight_init}. Use 'xavier_normal' or 'small_random'.")

        # バイアス
        self.v_bias = nn.Parameter(torch.zeros(n_visible))  # 可視層バイアス (b)
        self.h_bias = nn.Parameter(torch.zeros(n_hidden))   # 隠れ層バイアス (c)

        # 標準偏差 σ
        if learn_sigma:
            # log(σ)を学習して、σ = exp(log_sigma) で正の値を保証
            self.log_sigma = nn.Parameter(torch.ones(n_visible) * np.log(sigma_init))
        else:
            # 固定値として登録
            self.register_buffer('sigma', torch.ones(n_visible) * sigma_init)

    @property
    def sigma_sq(self):
        """σ^2 を返す"""
        if self.learn_sigma:
            sigma = torch.exp(self.log_sigma)
            return sigma ** 2
        else:
            return self.sigma ** 2

    def sample_h_given_v(self, v):
        """
        P(h=1|v) を計算し、サンプリング

        P(h_j = 1 | v) = sigmoid(c_j + Σ_i (v_i / σ_i^2) W_{ij})
        """
        # v / σ^2
        v_scaled = v / self.sigma_sq

        # 活性化: c + (v / σ^2) @ W
        activation = torch.matmul(v_scaled, self.W) + self.h_bias
        prob = torch.sigmoid(activation)

        return torch.bernoulli(prob), prob

    def sample_v_given_h(self, h, add_noise=True):
        """
        P(v|h) を計算し、サンプリング

        v | h ~ N(μ, σ^2) where μ_i = b_i + σ_i^2 * Σ_j W_{ij} h_j

        Note: 学習時はノイズなし（平均値のみ）で十分なことが多い
        """
        # 平均: b + σ^2 * (h @ W^T)
        # ただし、標準的な実装では b + h @ W^T を使うことも多い
        # ここでは Cho et al. (2011) に従い、σ^2 でスケーリング
        mean = self.v_bias + torch.matmul(h, self.W.t())

        if add_noise:
            # ガウスノイズを追加
            if self.learn_sigma:
                sigma = torch.exp(self.log_sigma)
            else:
                sigma = self.sigma
            noise = torch.randn_like(mean) * sigma
            sample = mean + noise
        else:
            sample = mean

        return sample, mean

    def free_energy(self, v):
        """
        自由エネルギー F(v) = -log Σ_h exp(-E(v,h))

        F(v) = Σ_i (v_i - b_i)^2 / (2σ_i^2) - Σ_j softplus(c_j + Σ_i (v_i / σ_i^2) W_{ij})
        """
        # 可視層項: Σ_i (v_i - b_i)^2 / (2σ_i^2)
        v_centered = v - self.v_bias
        visible_term = torch.sum((v_centered ** 2) / (2 * self.sigma_sq), dim=1)

        # 隠れ層項: Σ_j softplus(c_j + (v / σ^2) @ W)
        v_scaled = v / self.sigma_sq
        wx_b = torch.matmul(v_scaled, self.W) + self.h_bias
        hidden_term = torch.sum(F.softplus(wx_b), dim=1)

        return visible_term - hidden_term

    def train_step(self, v_pos, optimizer, pcd_chain, k_steps=1, add_noise=False):
        """
        PCD法による学習

        Args:
            v_pos: 正例データ (Batch, n_visible)
            optimizer: PyTorch optimizer
            pcd_chain: PCDバッファ（前回のnegative sample）
            k_steps: Gibbsサンプリングのステップ数
            add_noise: Trueの場合、v再構成時にノイズを追加

        Returns:
            loss: 自由エネルギー差
            v_neg: 更新後のPCDバッファ
        """
        # Negative Phase (Dreaming)
        v_neg = pcd_chain.detach()

        for _ in range(k_steps):
            # v -> h
            h_neg, _ = self.sample_h_given_v(v_neg)
            # h -> v（学習時はノイズなしが安定）
            v_neg, _ = self.sample_v_given_h(h_neg, add_noise=add_noise)

        # Free Energy Difference
        energy_pos = self.free_energy(v_pos).mean()
        energy_neg = self.free_energy(v_neg).mean()

        loss = energy_pos - energy_neg

        # Update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item(), v_neg.detach()

    def reconstruct(self, v, n_steps=1):
        """
        入力を再構成（デバッグ・評価用）

        Args:
            v: 入力データ
            n_steps: Gibbsステップ数

        Returns:
            v_recon: 再構成された可視層
        """
        with torch.no_grad():
            v_current = v
            for _ in range(n_steps):
                h_sample, _ = self.sample_h_given_v(v_current)
                v_current, _ = self.sample_v_given_h(h_sample, add_noise=False)
            return v_current

    def reconstruction_error(self, v):
        """
        再構成誤差（MSE）を計算
        """
        with torch.no_grad():
            v_recon = self.reconstruct(v, n_steps=1)
            mse = torch.mean((v - v_recon) ** 2)
            return mse.item()

    def get_latent_representation(self, v):
        """
        Adapterへの入力用：隠れ層の活性化確率を返す
        """
        _, h_prob = self.sample_h_given_v(v)
        return h_prob


if __name__ == '__main__':
    # デバイスの設定
    device = find_device()
    print(f"Using device: {device}")
    print()

    print("=" * 50)
    print("Testing RestrictedBoltzmannMachine (Bernoulli-Bernoulli)...")
    print("=" * 50)

    rbm = RestrictedBoltzmannMachine(n_visible=10, n_hidden=5).to(device)
    # 二値入力（0/1に量子化）
    v_test_binary = (torch.randn(32, 10) > 0).float().to(device)

    # Forward pass
    h_sample, h_prob = rbm.sample_h_given_v(v_test_binary)
    print(f"h_sample shape: {h_sample.shape}")
    print(f"h_prob range: [{h_prob.min():.3f}, {h_prob.max():.3f}]")
    print(f"h_sample unique values: {torch.unique(h_sample).tolist()}")

    # Reconstruction
    v_recon, v_prob = rbm.sample_v_given_h(h_sample)
    print(f"v_recon shape: {v_recon.shape}")
    print(f"v_recon unique values: {torch.unique(v_recon).tolist()}")

    # Free energy
    fe = rbm.free_energy(v_test_binary)
    print(f"Free energy shape: {fe.shape}, mean: {fe.mean():.3f}")

    # Latent representation
    latent = rbm.get_latent_representation(v_test_binary)
    print(f"Latent representation shape: {latent.shape}")

    print()
    print("=" * 50)
    print("Testing GaussianBernoulliRBM...")
    print("=" * 50)

    gbrbm = GaussianBernoulliRBM(n_visible=10, n_hidden=5).to(device)
    v_test_cont = torch.randn(32, 10).to(device)  # 連続値入力

    # Forward pass
    h_sample, h_prob = gbrbm.sample_h_given_v(v_test_cont)
    print(f"h_sample shape: {h_sample.shape}")
    print(f"h_prob range: [{h_prob.min():.3f}, {h_prob.max():.3f}]")

    # Reconstruction
    v_recon, v_mean = gbrbm.sample_v_given_h(h_sample)
    print(f"v_recon shape: {v_recon.shape}")
    print(f"v_recon range: [{v_recon.min():.3f}, {v_recon.max():.3f}]")

    # Free energy
    fe = gbrbm.free_energy(v_test_cont)
    print(f"Free energy shape: {fe.shape}, mean: {fe.mean():.3f}")

    # Reconstruction error
    mse = gbrbm.reconstruction_error(v_test_cont)
    print(f"Reconstruction MSE: {mse:.4f}")

    # Latent representation
    latent = gbrbm.get_latent_representation(v_test_cont)
    print(f"Latent representation shape: {latent.shape}")

    print()
    print("=" * 50)
    print("All tests passed!")
    print("=" * 50)