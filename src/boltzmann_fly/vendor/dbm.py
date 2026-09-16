import os, pickle, random, gc
from pathlib import Path
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
try:
    import matplotlib.pyplot as plt
except ImportError:  # boltzmann-fly: matplotlib is optional (only analyze_activations(plot=True) needs it)
    plt = None

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rbm import RestrictedBoltzmannMachine, GaussianBernoulliRBM
from .utils import find_device

class DeepBoltzmannMachine(nn.Module):
    """
    Deep Boltzmann Machine (DBM)

    多層のRBMを積み重ねた階層的な信念ネットワーク。
    Hinton et al. (2006) "A fast learning algorithm for deep belief nets" の実装。

    Architecture:
        v (visible) <-> h1 (hidden layer 1) <-> h2 (hidden layer 2) <-> ... <-> hN

    Training Procedure:
        1. Greedy Layer-wise Pretraining: 各層をRBMとして順次学習
        2. Joint Fine-tuning: 全層を同時に最適化（オプション）

    Gaussian-Bernoulli Mode:
        use_gbrbm=True の場合、可視層をガウス分布として扱う。
        これにより連続値入力（価格、頻度、beliefベクトル等）を直接扱える。
    """

    def __init__(self, layer_sizes, use_gbrbm=False, sigma_init=1.0, learn_sigma=False,
                 lag_window=0, J_comp_per_layer=None, mf_init=0.5):
        """
        Args:
            layer_sizes (list): 各層のユニット数 [n_visible, n_h1, n_h2, ...]
                               例: [35, 128, 64] → 可視層35次元、隠れ層1が128次元、隠れ層2が64次元
                               LR-DBMモードでもn_visibleは「元の観測次元」を指定する。
                               拡張入力次元は内部で自動計算される。
            use_gbrbm (bool): Trueの場合、最初の層をGaussian-Bernoulli RBMとして扱う
            sigma_init (float): GBRBMモードでのσの初期値
            learn_sigma (bool): Trueの場合、σを学習可能パラメータにする
            lag_window (int): LR-DBMのラグ窓幅 s（0の場合は通常DBM）
            J_comp_per_layer (list): 各隠れ層の圧縮次元数（Noneの場合は各層を1/4に圧縮）
            mf_init (float or list of float): Mean-field推論の各隠れ層の初期値。
                float: 全層同一の初期値（デフォルト: 0.5）
                list: 層ごとの初期値（例: [0.5, 0.1] で Layer1=0.5, Layer2=0.1）
                Layer2の初期値を下げることで、top-downフィードバックを弱め、
                Layer1のdead units発生を抑制できる。
        """
        super().__init__()

        assert len(layer_sizes) >= 2, "At least 2 layers (visible + 1 hidden) required"

        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes) - 1  # 隠れ層の数
        self.use_gbrbm = use_gbrbm
        self.learn_sigma = learn_sigma

        # Mean-field初期値の設定
        if isinstance(mf_init, (list, tuple)):
            assert len(mf_init) == self.n_layers, \
                f"mf_init length ({len(mf_init)}) must match n_layers ({self.n_layers})"
            self.mf_init = list(mf_init)
        else:
            self.mf_init = [float(mf_init)] * self.n_layers

        # LR-DBM（Latent-Recurrent DBM）パラメータ
        self.lag_window = lag_window
        if J_comp_per_layer is None:
            # デフォルト: 各隠れ層を1/4に圧縮
            self.J_comp_per_layer = [max(1, layer_sizes[i+1] // 4) for i in range(self.n_layers)]
        else:
            assert len(J_comp_per_layer) == self.n_layers, \
                f"J_comp_per_layer length ({len(J_comp_per_layer)}) must match n_layers ({self.n_layers})"
            self.J_comp_per_layer = J_comp_per_layer

        # 圧縮後の総次元数
        self.d_comp = sum(self.J_comp_per_layer)

        # 元の可視層次元（ラグ履歴を含まない純粋な観測次元）
        # layer_sizes[0]は常に「元の観測次元」を表す
        self.d_v_original = layer_sizes[0]

        # LR-DBMモードの場合、拡張入力次元を計算
        if lag_window > 0:
            self.d_v_extended = self.d_v_original + lag_window * self.d_comp
        else:
            self.d_v_extended = self.d_v_original

        # 各層間の重み行列とバイアスを保持
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()

        for i in range(self.n_layers):
            if i == 0:
                # 第1層: LR-DBMの場合は拡張入力次元を使用
                n_lower = self.d_v_extended
            else:
                n_lower = layer_sizes[i]
            n_upper = layer_sizes[i + 1]

            # 重み行列 W_i: (n_lower, n_upper)
            W = nn.Parameter(torch.randn(n_lower, n_upper) * 0.01)
            self.weights.append(W)

            # 下層のバイアス (最初の層は可視層のバイアス)
            if i == 0:
                # LR-DBMの場合は拡張入力次元でバイアスを初期化
                self.biases.append(nn.Parameter(torch.zeros(n_lower)))

            # 上層のバイアス
            self.biases.append(nn.Parameter(torch.zeros(n_upper)))

        # GBRBM用のσパラメータ
        if use_gbrbm:
            # LR-DBMの場合は拡張入力次元を使用
            n_visible = self.d_v_extended
            if learn_sigma:
                # log(σ)を学習して、σ = exp(log_sigma) で正の値を保証
                self.log_sigma = nn.Parameter(torch.ones(n_visible) * np.log(sigma_init))
            else:
                # 固定値として登録
                self.register_buffer('sigma', torch.ones(n_visible) * sigma_init)

        # Greedy Pretraining用のRBMリスト（学習後は不要だが、参照用に保持）
        self.rbm_stack = []

    @property
    def sigma_sq(self):
        """σ^2 を返す（GBRBMモードのみ）"""
        if not self.use_gbrbm:
            raise ValueError("sigma_sq is only available in GBRBM mode")
        if self.learn_sigma:
            sigma = torch.exp(self.log_sigma)
            return sigma ** 2
        else:
            return self.sigma ** 2

    def _get_sigma_sq_for_v(self, v: torch.Tensor) -> torch.Tensor:
        """sigma_sq を v の次元に合わせて返す（LR-DBM拡張入力対応）。
        v が sigma_sq より狭い場合は先頭部分のみ使用、
        v が sigma_sq より広い場合は 1.0 でパディング。"""
        sq = self.sigma_sq
        v_dim = v.shape[-1]
        if v_dim < sq.shape[0]:
            sq = sq[:v_dim]
        elif v_dim > sq.shape[0]:
            pad = torch.ones(v_dim - sq.shape[0], device=sq.device, dtype=sq.dtype)
            sq = torch.cat([sq, pad])
        return sq

    def initialize_from_rbms(self, rbm_list, weight_scaling=None):
        """
        事前学習済みのRBMスタックから重みをコピーする

        Hinton et al. (2009) "Deep Boltzmann Machines" の手法に従い、
        中間層の重みをスケーリングする。
        理由: 中間層はbottom-upとtop-downの両方から入力を受けるため、
        合計入力が約2倍になることを補正する。

        Args:
            rbm_list (list): 学習済みRBMのリスト
            weight_scaling (list of float or None): 各層の重みスケーリング係数。
                Noneの場合はデフォルト（first/last=1.0, middle=0.5）。
                例: [1.0, 0.5, 1.0] で3層DBMの中間層のみ0.5倍。
        """
        assert len(rbm_list) == self.n_layers, \
            f"Expected {self.n_layers} RBMs, got {len(rbm_list)}"

        # デフォルトのスケーリング係数を決定
        if weight_scaling is None:
            weight_scaling = []
            for i in range(self.n_layers):
                if i == 0 or i == self.n_layers - 1:
                    weight_scaling.append(1.0)
                else:
                    weight_scaling.append(0.5)
        assert len(weight_scaling) == self.n_layers, \
            f"weight_scaling length ({len(weight_scaling)}) must match n_layers ({self.n_layers})"

        for i, rbm in enumerate(rbm_list):
            scale = weight_scaling[i]

            # 重みをコピー（スケーリング適用）
            if i == 0 and self.lag_window > 0:
                # LR-DBM: RBMは元の可視層次元のみで学習済み
                # 元次元部分はRBMからコピー、履歴部分はXavier初期化
                n_v_orig = self.d_v_original
                self.weights[i].data[:n_v_orig, :].copy_(rbm.W.data * scale)
                nn.init.xavier_normal_(self.weights[i].data[n_v_orig:, :])
            else:
                self.weights[i].data.copy_(rbm.W.data * scale)

            # バイアスをコピー（そのまま）
            if i == 0 and self.lag_window > 0:
                # LR-DBM: 元次元部分のみコピー、履歴部分はゼロ
                n_v_orig = self.d_v_original
                self.biases[0].data[:n_v_orig].copy_(rbm.v_bias.data)
                self.biases[0].data[n_v_orig:] = 0.0

                # GBRBMの場合、σの元次元部分をコピー（履歴部分はデフォルト値を維持）
                if self.use_gbrbm and isinstance(rbm, GaussianBernoulliRBM):
                    if self.learn_sigma and rbm.learn_sigma:
                        self.log_sigma.data[:n_v_orig].copy_(rbm.log_sigma.data)
                    elif not self.learn_sigma and not rbm.learn_sigma:
                        self.sigma.data[:n_v_orig].copy_(rbm.sigma.data)
                    elif self.learn_sigma and not rbm.learn_sigma:
                        self.log_sigma.data[:n_v_orig].copy_(torch.log(rbm.sigma.data))
                    else:
                        self.sigma.data[:n_v_orig].copy_(torch.exp(rbm.log_sigma.data))
            elif i == 0:
                self.biases[0].data.copy_(rbm.v_bias.data)

                # GBRBMの場合、σもコピー
                if self.use_gbrbm and isinstance(rbm, GaussianBernoulliRBM):
                    if self.learn_sigma and rbm.learn_sigma:
                        self.log_sigma.data.copy_(rbm.log_sigma.data)
                    elif not self.learn_sigma and not rbm.learn_sigma:
                        self.sigma.data.copy_(rbm.sigma.data)
                    elif self.learn_sigma and not rbm.learn_sigma:
                        self.log_sigma.data.copy_(torch.log(rbm.sigma.data))
                    else:  # not self.learn_sigma and rbm.learn_sigma
                        self.sigma.data.copy_(torch.exp(rbm.log_sigma.data))

            self.biases[i + 1].data.copy_(rbm.h_bias.data)

        self.rbm_stack = rbm_list
        mode_str = "GBRBM + BB-RBMs" if self.use_gbrbm else "BB-RBMs"
        print(f"✅ Initialized DBM from {len(rbm_list)} pretrained RBMs ({mode_str})")
        if self.lag_window > 0:
            print(f"   (履歴部分の重みはXavier初期化、元次元部分はRBMからコピー)")
        scale_str = ", ".join([f"L{i+1}={s:.2f}" for i, s in enumerate(weight_scaling)])
        print(f"   (Weight scaling: {scale_str})")

    def compensate_biases(self, data, device, batch_size=256):
        """
        RBM→DBM変換時のバイアス補正。

        DBMでは中間層がtop-downフィードバックを受けるため、
        RBMのバイアスのままだと活性化の均衡が崩れてdead unitsが発生する。

        各非最上層のバイアスから E[W_{k+1}^T @ h_{k+1}] を減算して補正する。
        E[h_{k+1}] はデータをbottom-up（RBM的）に通して推定する。

        Args:
            data: 入力データ (N, n_visible) — pretraining時に使用したデータ
            device: torch.device
            batch_size: バッチサイズ
        """
        if self.n_layers <= 1:
            print("   (Bias compensation skipped: single hidden layer, no top-down)")
            return

        self.eval()

        # Bottom-up propagation（RBM的な前向き推論）でE[h_k]を推定
        all_h_means = [[] for _ in range(self.n_layers)]

        with torch.no_grad():
            for i in range(0, len(data), batch_size):
                batch = data[i:i + batch_size].to(device)

                # 可視層 → h1
                if self.use_gbrbm:
                    v_scaled = batch / self._get_sigma_sq_for_v(batch)
                else:
                    v_scaled = batch
                activation = torch.matmul(v_scaled, self.weights[0]) + self.biases[1]
                h_prob = torch.sigmoid(activation)
                all_h_means[0].append(h_prob.cpu())

                # h1 → h2 → ... → hN（bottom-upのみ）
                for layer_idx in range(1, self.n_layers):
                    activation = torch.matmul(h_prob, self.weights[layer_idx]) + self.biases[layer_idx + 1]
                    h_prob = torch.sigmoid(activation)
                    all_h_means[layer_idx].append(h_prob.cpu())

        # E[h_k]を計算
        expected_h = []
        for layer_idx in range(self.n_layers):
            h_all = torch.cat(all_h_means[layer_idx], dim=0)
            expected_h.append(h_all.mean(dim=0))

        # 非最上層のバイアスを補正
        print(f"   Bias compensation:")
        for layer_idx in range(self.n_layers - 1):
            # Layer layer_idx は layer_idx+1 からのtop-downを受ける
            # top_down = h_{k+1} @ W_{k+1}^T  (mean_field_inferenceと同じ計算)
            # 補正: bias_{k} -= E[h_{k+1}] @ W_{k+1}^T
            e_h_above = expected_h[layer_idx + 1].to(device)
            correction = torch.matmul(e_h_above, self.weights[layer_idx + 1].t())

            before_mean = self.biases[layer_idx + 1].data.mean().item()
            self.biases[layer_idx + 1].data -= correction
            after_mean = self.biases[layer_idx + 1].data.mean().item()

            print(f"     L{layer_idx+1}: E[h{layer_idx+2}]={e_h_above.mean():.4f}, "
                  f"correction norm={correction.norm():.4f}, "
                  f"bias mean {before_mean:.4f} → {after_mean:.4f}")

    def mean_field_inference(self, v, n_iter=10):
        """
        Mean-field近似による推論

        DBMでは中間層が上下両方から入力を受けるため、
        exact inferenceは困難。Mean-field approximationで近似する。

        Args:
            v: 可視層の入力 (Batch, n_visible)
            n_iter: 平均場反復回数

        Returns:
            h_probs (list): 各隠れ層の活性化確率 [h1_prob, h2_prob, ...]
        """
        batch_size = v.shape[0]

        # 各層の確率値を初期化（self.mf_initで層ごとに設定可能）
        h_probs = []
        for i in range(self.n_layers):
            h_prob = torch.ones(batch_size, self.layer_sizes[i + 1], device=v.device) * self.mf_init[i]
            h_probs.append(h_prob)

        # GBRBMモードの場合、可視層入力をσ^2でスケーリング
        if self.use_gbrbm:
            v_scaled = v / self._get_sigma_sq_for_v(v)
        else:
            v_scaled = v

        # Mean-field iterations
        for _ in range(n_iter):
            # Bottom-up: 可視層から隠れ層1へ
            bottom_up = torch.matmul(v_scaled, self.weights[0]) + self.biases[1]

            if self.n_layers > 1:
                # Top-down: 隠れ層2から隠れ層1へ
                top_down = torch.matmul(h_probs[1], self.weights[1].t())
                h_probs[0] = torch.sigmoid(bottom_up + top_down)
            else:
                h_probs[0] = torch.sigmoid(bottom_up)

            # 中間層の更新（存在する場合）
            for i in range(1, self.n_layers - 1):
                bottom_up = torch.matmul(h_probs[i - 1], self.weights[i]) + self.biases[i + 1]
                top_down = torch.matmul(h_probs[i + 1], self.weights[i + 1].t())
                h_probs[i] = torch.sigmoid(bottom_up + top_down)

            # 最上層の更新
            if self.n_layers > 1:
                bottom_up = torch.matmul(h_probs[-2], self.weights[-1]) + self.biases[-1]
                h_probs[-1] = torch.sigmoid(bottom_up)

        return h_probs

    def sample_h_given_v(self, v, n_iter=10):
        """
        可視層vが与えられたときの隠れ層のサンプリング

        Args:
            v: 可視層 (Batch, n_visible)
            n_iter: Mean-field反復回数

        Returns:
            h_samples (list): 各隠れ層のサンプル [h1, h2, ...]
            h_probs (list): 各隠れ層の確率 [h1_prob, h2_prob, ...]
        """
        h_probs = self.mean_field_inference(v, n_iter)

        # 確率に基づいてサンプリング
        h_samples = [torch.bernoulli(prob) for prob in h_probs]

        return h_samples, h_probs

    def sample_v_given_h(self, h_list, add_noise=True):
        """
        隠れ層が与えられたときの可視層のサンプリング

        Args:
            h_list (list): 各隠れ層の状態 [h1, h2, ...]
            add_noise (bool): GBRBMモードでガウスノイズを追加するかどうか

        Returns:
            v_sample: 可視層のサンプル (Batch, n_visible)
            v_prob_or_mean: Bernoulliモードでは確率、GBRBMモードでは平均
        """
        if self.use_gbrbm:
            # GBRBMモード: ガウス分布からサンプリング
            # 平均: b + h @ W^T
            mean = self.biases[0] + torch.matmul(h_list[0], self.weights[0].t())

            if add_noise:
                # ガウスノイズを追加
                if self.learn_sigma:
                    sigma = torch.exp(self.log_sigma)
                else:
                    sigma = self.sigma
                noise = torch.randn_like(mean) * sigma
                v_sample = mean + noise
            else:
                v_sample = mean

            return v_sample, mean
        else:
            # Bernoulliモード: 二値サンプリング
            activation = torch.matmul(h_list[0], self.weights[0].t()) + self.biases[0]
            v_prob = torch.sigmoid(activation)
            v_sample = torch.bernoulli(v_prob)

            return v_sample, v_prob

    def gibbs_sampling(self, v_init, k_steps=1, n_iter=10):
        """
        Gibbs sampling for DBM

        Args:
            v_init: 初期可視層 (Batch, n_visible)
            k_steps: Gibbsステップ数
            n_iter: 各ステップでのMean-field反復回数

        Returns:
            v_sample: サンプリング後の可視層
            h_samples: サンプリング後の隠れ層リスト
        """
        v = v_init

        for _ in range(k_steps):
            # v -> h (all layers)
            h_samples, _ = self.sample_h_given_v(v, n_iter)
            # h -> v
            v, _ = self.sample_v_given_h(h_samples)

        return v, h_samples

    def free_energy(self, v, n_iter=10):
        """
        自由エネルギーの計算（変分近似）

        DBMの正確な自由エネルギーは計算困難なため、
        Mean-field近似を使った変分自由エネルギーを使用。

        重要: Mean-field inferenceの計算グラフをdetachして、
        AutoGradが反復ループを遡らないようにする。
        これにより勾配計算のコストを削減し、数値安定性を向上させる。

        Args:
            v: 可視層 (Batch, n_visible)
            n_iter: Mean-field反復回数（推奨: 5-10）

        Returns:
            F(v): 自由エネルギー (Batch,)
        """
        # Mean-field inference（勾配を切る）
        # Positive phaseでは、h_probsは固定値として扱う
        with torch.no_grad():
            h_probs_detached = self.mean_field_inference(v, n_iter)

        # 勾配計算が必要な部分は、detachしたh_probsを使って再計算
        # これにより、重みに対する勾配は正しく計算されるが、
        # Mean-fieldループへの勾配伝播は避けられる
        h_probs = [h.detach() for h in h_probs_detached]

        # 可視層の寄与
        if self.use_gbrbm:
            # GBRBMモード: (v - b)^2 / (2σ^2)
            sq = self._get_sigma_sq_for_v(v)
            v_centered = v - self.biases[0]
            v_term = torch.sum((v_centered ** 2) / (2 * sq), dim=1)
            # v_scaledを使って隠れ層への入力を計算
            v_scaled = v / sq
        else:
            # Bernoulliモード: v @ b
            v_term = torch.matmul(v, self.biases[0])
            v_scaled = v

        # 各隠れ層の寄与
        hidden_term = 0.0

        # 第1層
        wx_b = torch.matmul(v_scaled, self.weights[0]) + self.biases[1]
        if self.n_layers > 1:
            wx_b += torch.matmul(h_probs[1], self.weights[1].t())
        hidden_term += torch.sum(h_probs[0] * wx_b, dim=1)

        # 中間層
        for i in range(1, self.n_layers - 1):
            wx_b = torch.matmul(h_probs[i - 1], self.weights[i]) + self.biases[i + 1]
            wx_b += torch.matmul(h_probs[i + 1], self.weights[i + 1].t())
            hidden_term += torch.sum(h_probs[i] * wx_b, dim=1)

        # 最上層
        if self.n_layers > 1:
            wx_b = torch.matmul(h_probs[-2], self.weights[-1]) + self.biases[-1]
            hidden_term += torch.sum(h_probs[-1] * wx_b, dim=1)

        if self.use_gbrbm:
            # GBRBMモード: v_termは正（二乗項）、hidden_termは負にしたい
            return v_term - hidden_term
        else:
            return -v_term - hidden_term

    def train_step(self, v_pos, optimizer, pcd_buffer, k_steps=5, n_iter=10,
                    grad_clip=None):
        """
        PCD法によるDBMの学習

        Args:
            v_pos: Positive phase の可視層データ (Batch, n_visible)
            optimizer: PyTorch optimizer
            pcd_buffer: PCDバッファ（前回の夢の状態）
            k_steps: Gibbsサンプリングのステップ数
            n_iter: Mean-field反復回数
            grad_clip: 勾配クリッピングの最大ノルム（Noneで無効）

        Returns:
            loss: 自由エネルギー差
            v_neg: 更新後のPCDバッファ
        """
        # Negative phase (Dreaming)
        v_neg, _ = self.gibbs_sampling(pcd_buffer, k_steps, n_iter)

        # Free energy calculation
        energy_pos = self.free_energy(v_pos, n_iter).mean()
        energy_neg = self.free_energy(v_neg, n_iter).mean()

        loss = energy_pos - energy_neg

        # Update
        optimizer.zero_grad()
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip)
        optimizer.step()

        return loss.item(), v_neg.detach()

    def get_latent_representation(self, v, n_iter=10, use_top_layer=False):
        """
        Adapterへの入力用：可視層から潜在表現を取得

        Args:
            v: 可視層 (Batch, n_visible)
            n_iter: Mean-field反復回数
            use_top_layer: Trueなら最上層を返す、Falseなら全層を結合

        Returns:
            潜在表現 (Batch, latent_dim)
        """
        _, h_probs = self.sample_h_given_v(v, n_iter)

        if use_top_layer:
            # 最上層のみを使用（より抽象的な表現）
            return h_probs[-1]
        else:
            # 全層を結合（より情報量の多い表現）
            return torch.cat(h_probs, dim=1)

    # ==========================================
    # LR-DBM (Latent-Recurrent DBM) Methods
    # ==========================================

    def compress(self, h_layers):
        """
        全隠れ層の活性化を圧縮表現に変換する（LR-DBM用）

        各層をチャンク平均で圧縮する固定的手続き（パラメータなし）。
        J_l が J_comp_l で割り切れない場合は、余りのユニットを切り捨てる。

        Args:
            h_layers: List[Tensor] - 各層の隠れユニット活性化確率
                      h_layers[l].shape = (batch, J_l)

        Returns:
            compressed: Tensor - 圧縮された表現
                       shape = (batch, d_comp) where d_comp = Σ_l J_comp^l
        """
        compressed_list = []

        for l, h_l in enumerate(h_layers):
            batch_size, J_l = h_l.shape
            J_comp_l = self.J_comp_per_layer[l]
            chunk_size = J_l // J_comp_l

            # チャンク平均: (batch, J_l) → (batch, J_comp_l)
            # 余りがある場合は切り捨て
            h_l_truncated = h_l[:, :J_comp_l * chunk_size]
            h_l_reshaped = h_l_truncated.reshape(batch_size, J_comp_l, chunk_size)
            compressed_l = h_l_reshaped.mean(dim=-1)
            compressed_list.append(compressed_l)

        # 全層を結合
        return torch.cat(compressed_list, dim=-1)

    def prepare_history(self, history_queue, device=None):
        """
        履歴キューから圧縮表現のリストを準備する（LR-DBM用）

        履歴が lag_window より少ない場合はゼロパディングを行う。

        Args:
            history_queue: deque - 過去の圧縮表現を保持するキュー
                          各要素は shape = (batch, d_comp)
            device: torch.device - デバイス（ゼロパディング用）

        Returns:
            history_list: List[Tensor] - lag_window個の圧縮表現リスト
                         最新のものが先頭（時系列順: [c(t-1), c(t-2), ..., c(t-s)]）
        """
        if self.lag_window == 0:
            return []

        history_list = list(history_queue)

        # ゼロパディングが必要な場合
        n_available = len(history_list)
        n_padding = self.lag_window - n_available

        if n_padding > 0:
            if n_available > 0:
                # 既存の履歴からshapeを取得
                batch_size = history_list[0].shape[0]
                target_device = history_list[0].device
            else:
                # 履歴が空の場合、引数からdeviceを取得
                batch_size = 1  # 後で実際のバッチサイズに合わせる
                target_device = device if device is not None else torch.device('cpu')

            # ゼロパディング
            padding = [torch.zeros(batch_size, self.d_comp, device=target_device) for _ in range(n_padding)]
            history_list = padding + history_list

        # 最新のものを先頭にする（history_queueは古い順なので逆順に）
        return list(reversed(history_list))

    def build_extended_input(self, v_t, history_list):
        """
        現在の観測と履歴を結合して拡張入力を作成する（LR-DBM用）

        Args:
            v_t: Tensor - 現在の観測 shape = (batch, d_v_original)
            history_list: List[Tensor] - 圧縮履歴リスト [c(t-1), c(t-2), ..., c(t-s)]
                         各要素は shape = (batch, d_comp)

        Returns:
            v_bar: Tensor - 拡張入力 shape = (batch, d_v_original + lag_window * d_comp)
        """
        if self.lag_window == 0 or len(history_list) == 0:
            return v_t

        # パディングのバッチサイズを実際の入力に合わせる
        batch_size = v_t.shape[0]
        adjusted_history = []
        for h in history_list:
            if h.shape[0] != batch_size:
                # バッチサイズが異なる場合は拡張/切り捨て
                if h.shape[0] == 1:
                    h = h.expand(batch_size, -1)
                else:
                    h = h[:batch_size]
            adjusted_history.append(h)

        # v(t) と履歴を結合: [v(t); c(t-1); c(t-2); ...; c(t-s)]
        return torch.cat([v_t] + adjusted_history, dim=-1)

    def get_all_hidden_probs(self, v, n_iter=10):
        """
        可視層から全隠れ層の活性化確率を取得する（compress用）

        Args:
            v: 可視層 (Batch, n_visible)
            n_iter: Mean-field反復回数

        Returns:
            h_probs: List[Tensor] - 各隠れ層の活性化確率
        """
        return self.mean_field_inference(v, n_iter)

    def analyze_activations(
        self,
        dataloader,
        device,
        n_iter: int = 10,
        plot: bool = True,
        figsize: tuple = (14, 10)
    ) -> dict:
        """
        隠れ層の活性化パターンを分析する

        各隠れユニットについて:
        - 平均活性化率（データ全体での活性化頻度）
        - ユニット間の活性化率の分布（ばらつき）
        - ユニットごとの分散（常に同じ値を出力していないか）

        Args:
            dataloader: データローダー（'gbm_vector'キーを持つ辞書を返す）
            device: torch.device
            n_iter: Mean-field反復回数
            plot: Trueの場合、可視化を行う
            figsize: プロットのサイズ

        Returns:
            dict: 各層の活性化統計量を含む辞書
        """
        self.eval()

        # 全データの隠れ層活性化を収集
        all_h_probs = [[] for _ in range(self.n_layers)]

        with torch.no_grad():
            for batch in dataloader:
                v = batch['gbm_vector'].to(device)
                h_probs = self.mean_field_inference(v, n_iter)

                for i, h in enumerate(h_probs):
                    all_h_probs[i].append(h.cpu())

        # 統計量の計算
        results = {}

        for layer_idx in range(self.n_layers):
            h_all = torch.cat(all_h_probs[layer_idx], dim=0)  # (N, n_hidden)
            n_samples, n_units = h_all.shape

            # 各ユニットの統計量
            mean_activation = h_all.mean(dim=0).numpy()  # ユニットごとの平均活性化率
            std_per_unit = h_all.std(dim=0).numpy()      # ユニットごとの標準偏差
            var_per_unit = h_all.var(dim=0).numpy()      # ユニットごとの分散

            # 全体統計
            overall_mean = mean_activation.mean()
            overall_std = mean_activation.std()

            # Dead/Saturated units の検出
            dead_threshold = 0.01
            saturated_threshold = 0.99
            dead_units = (mean_activation < dead_threshold).sum()
            saturated_units = (mean_activation > saturated_threshold).sum()

            # 低分散ユニット（常に同じ値を出力）
            low_var_threshold = 0.01
            low_var_units = (var_per_unit < low_var_threshold).sum()

            results[f"layer_{layer_idx + 1}"] = {
                "n_units": n_units,
                "n_samples": n_samples,
                "mean_activation": mean_activation,
                "std_per_unit": std_per_unit,
                "var_per_unit": var_per_unit,
                "overall_mean": overall_mean,
                "overall_std": overall_std,
                "dead_units": int(dead_units),
                "saturated_units": int(saturated_units),
                "low_var_units": int(low_var_units),
            }

        # サマリ出力
        print("\n" + "=" * 60)
        print("HIDDEN LAYER ACTIVATION ANALYSIS")
        print("=" * 60)

        for layer_idx in range(self.n_layers):
            layer_key = f"layer_{layer_idx + 1}"
            r = results[layer_key]
            print(f"\n📊 Layer {layer_idx + 1} ({r['n_units']} units, {r['n_samples']} samples):")
            print(f"   Mean activation rate: {r['overall_mean']:.4f} ± {r['overall_std']:.4f}")
            print(f"   Dead units (<{dead_threshold}): {r['dead_units']} ({r['dead_units']/r['n_units']*100:.1f}%)")
            print(f"   Saturated units (>{saturated_threshold}): {r['saturated_units']} ({r['saturated_units']/r['n_units']*100:.1f}%)")
            print(f"   Low variance units: {r['low_var_units']} ({r['low_var_units']/r['n_units']*100:.1f}%)")

        # 可視化
        if plot:
            n_rows = self.n_layers
            _, axes = plt.subplots(n_rows, 3, figsize=figsize)
            if n_rows == 1:
                axes = axes.reshape(1, -1)

            for layer_idx in range(self.n_layers):
                layer_key = f"layer_{layer_idx + 1}"
                r = results[layer_key]

                # 1. 平均活性化率のヒストグラム
                ax1 = axes[layer_idx, 0]
                ax1.hist(r["mean_activation"], bins=50, edgecolor='black', alpha=0.7)
                ax1.axvline(r["overall_mean"], color='red', linestyle='--',
                           label=f'Mean: {r["overall_mean"]:.3f}')
                ax1.set_xlabel("Mean Activation Rate")
                ax1.set_ylabel("Count")
                ax1.set_title(f"Layer {layer_idx + 1}: Activation Distribution")
                ax1.legend()

                # 2. ユニットごとの標準偏差
                ax2 = axes[layer_idx, 1]
                ax2.hist(r["std_per_unit"], bins=50, edgecolor='black', alpha=0.7, color='orange')
                ax2.set_xlabel("Std per Unit")
                ax2.set_ylabel("Count")
                ax2.set_title(f"Layer {layer_idx + 1}: Per-unit Std Distribution")

                # 3. 平均活性化率（ソート済み）
                ax3 = axes[layer_idx, 2]
                sorted_mean = np.sort(r["mean_activation"])
                ax3.plot(sorted_mean, linewidth=0.8)
                ax3.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
                ax3.fill_between(range(len(sorted_mean)), 0, sorted_mean, alpha=0.3)
                ax3.set_xlabel("Unit Index (sorted)")
                ax3.set_ylabel("Mean Activation")
                ax3.set_title(f"Layer {layer_idx + 1}: Sorted Activation Rates")
                ax3.set_ylim(0, 1)

            plt.tight_layout()
            plt.show()

        return results

    def get_activation_summary(
        self,
        dataloader,
        device,
        n_iter: int = 10
    ) -> pd.DataFrame:
        """
        活性化分析の結果をDataFrameで返す（プロットなし）

        Args:
            dataloader: データローダー
            device: torch.device
            n_iter: Mean-field反復回数

        Returns:
            pd.DataFrame: 各層の統計サマリ
        """
        results = self.analyze_activations(
            dataloader, device, n_iter, plot=False
        )

        summary_data = []
        for layer_idx in range(self.n_layers):
            layer_key = f"layer_{layer_idx + 1}"
            r = results[layer_key]
            summary_data.append({
                "Layer": layer_idx + 1,
                "Units": r["n_units"],
                "Mean Act.": f"{r['overall_mean']:.4f}",
                "Std Act.": f"{r['overall_std']:.4f}",
                "Dead (%)": f"{r['dead_units']/r['n_units']*100:.1f}",
                "Saturated (%)": f"{r['saturated_units']/r['n_units']*100:.1f}",
                "Low Var (%)": f"{r['low_var_units']/r['n_units']*100:.1f}",
            })

        return pd.DataFrame(summary_data)


def greedy_pretrain_dbm(layer_sizes, dataloader, device,
                        n_epochs_per_layer=100, lr=0.01, k_steps=5,
                        pcd_batch_size=128, use_gbrbm=False,
                        sigma_init=1.0, learn_sigma=False,
                        lag_window=0, J_comp_per_layer=None,
                        weight_init="xavier_normal",
                        weight_scaling=None,
                        mf_init=0.5,
                        pretraining_weight_decay=1e-3,
                        compensate_biases=True,
                        use_wandb=False,
                        verbose=2):
    """
    Greedy Layer-wise Pretraining for DBM

    各層を順次RBMとして学習し、最後にDBMを構築する。

    Args:
        layer_sizes (list): 各層のサイズ [n_visible, n_h1, n_h2, ...]
        dataloader: データローダー（'gbm_vector'キーを持つ辞書を返す想定）
        device: torch.device
        n_epochs_per_layer: 各層の学習エポック数
        lr: 学習率
        k_steps: PCDステップ数
        pcd_batch_size: PCDバッファのバッチサイズ
        use_gbrbm (bool): Trueの場合、最初の層をGaussian-Bernoulli RBMとして学習
        sigma_init (float): GBRBMモードでのσの初期値
        learn_sigma (bool): Trueの場合、σを学習可能パラメータにする
        lag_window (int): LR-DBMのラグ窓幅 s（0の場合は通常DBM）
        J_comp_per_layer (list): 各隠れ層の圧縮次元数

    Returns:
        dbm: 初期化されたDeepBoltzmannMachine
        rbm_list: 学習済みRBMのリスト
    """
    # layer_sizes[0]がNoneの場合、dataloaderから自動推論
    if layer_sizes[0] is None:
        sample_batch = next(iter(dataloader))
        n_visible_auto = sample_batch['gbm_vector'].shape[1]
        layer_sizes = layer_sizes.copy()  # 元のリストを変更しないようにコピー
        layer_sizes[0] = n_visible_auto
        print(f"⚙️  Auto-detected n_visible from dataloader: {n_visible_auto}")

    n_layers = len(layer_sizes) - 1
    rbm_list = []

    mode_str = "GBRBM + BB-RBMs" if use_gbrbm else "BB-RBMs"
    print(f"🚀 Starting Greedy Layer-wise Pretraining for {n_layers}-layer DBM ({mode_str})")
    print(f"   Architecture: {' -> '.join(map(str, layer_sizes))}")

    # 現在の入力データ（最初は可視層データ）
    current_data = None
    original_data = None  # バイアス補正用に元の可視層データを保持

    for layer_idx in range(n_layers):
        n_visible = layer_sizes[layer_idx]
        n_hidden = layer_sizes[layer_idx + 1]

        print(f"\n{'='*60}")

        # 第1層でGBRBMを使用するかどうか
        use_gbrbm_for_this_layer = (layer_idx == 0 and use_gbrbm)

        if use_gbrbm_for_this_layer:
            print(f"📚 Layer {layer_idx + 1}/{n_layers}: GBRBM({n_visible} -> {n_hidden})")
            print(f"{'='*60}")
            # GaussianBernoulliRBMを初期化
            rbm = GaussianBernoulliRBM(n_visible, n_hidden,
                                       learn_sigma=learn_sigma,
                                       sigma_init=sigma_init,
                                       weight_init=weight_init).to(device)
        else:
            print(f"📚 Layer {layer_idx + 1}/{n_layers}: RBM({n_visible} -> {n_hidden})")
            print(f"{'='*60}")
            # RestrictedBoltzmannMachineを初期化
            rbm = RestrictedBoltzmannMachine(n_visible, n_hidden).to(device)

        # バイアス初期化（第1層のみデータ統計を使用）
        if layer_idx == 0:
            # データローダーから全データを取得
            all_vectors = []
            for batch in dataloader:
                all_vectors.append(batch['gbm_vector'])
            all_vectors = torch.cat(all_vectors, dim=0)

            if use_gbrbm_for_this_layer:
                # GBRBMの場合: バイアスをデータの平均に設定
                v_mean = all_vectors.mean(dim=0).to(device)
                rbm.v_bias.data = v_mean
                # σはデータの標準偏差から初期化することも可能
                # rbm.sigma.data = all_vectors.std(dim=0).to(device)
            else:
                # BB-RBMの場合: ロジットでバイアス初期化
                v_mean = all_vectors.mean(dim=0).to(device)
                eps = 1e-4
                v_mean = torch.clamp(v_mean, eps, 1.0 - eps)
                rbm.v_bias.data = torch.log(v_mean / (1.0 - v_mean))

            current_data = all_vectors
            original_data = all_vectors  # バイアス補正用

        # current_dataをdeviceに事前転送（毎バッチのCPU→GPU転送を回避）
        current_data = current_data.to(device)

        # Optimizer
        optimizer = torch.optim.Adam(rbm.parameters(), lr=lr, weight_decay=pretraining_weight_decay)

        # PCDバッファ
        if use_gbrbm_for_this_layer:
            # GBRBMの場合: ガウス分布からサンプリング
            pcd_buffer = torch.randn(pcd_batch_size, n_visible).to(device)
        else:
            # BB-RBMの場合: ベルヌーイ分布からサンプリング
            pcd_buffer = torch.bernoulli(torch.rand(pcd_batch_size, n_visible)).to(device)

        # 学習ループ
        rbm.train()
        for epoch in tqdm(range(n_epochs_per_layer), desc=f"{'GBRBM' if use_gbrbm_for_this_layer else 'RBM'} Layer {layer_idx + 1}", disable=(verbose < 2)):
            total_loss = 0
            total_recon_loss = 0
            n_batches = 0

            # バッチごとの学習
            for i in range(0, len(current_data), pcd_batch_size):
                batch_data = current_data[i:i + pcd_batch_size].to(device)

                if batch_data.shape[0] < pcd_batch_size:
                    continue  # 最後の不完全なバッチはスキップ

                # Train step
                loss_val, v_neg = rbm.train_step(batch_data, optimizer, pcd_buffer, k_steps)
                pcd_buffer = v_neg.detach()

                # Reconstruction Loss
                with torch.no_grad():
                    _, h_prob = rbm.sample_h_given_v(batch_data)
                    v_recon, _ = rbm.sample_v_given_h(h_prob)

                    if use_gbrbm_for_this_layer:
                        # GBRBMの場合: MSE
                        recon_loss = torch.mean((batch_data - v_recon) ** 2)
                    else:
                        # BB-RBMの場合: Binary Cross Entropy
                        eps = 1e-7
                        v_recon = torch.clamp(v_recon, eps, 1.0 - eps)
                        recon_loss = -torch.mean(
                            batch_data * torch.log(v_recon) +
                            (1 - batch_data) * torch.log(1 - v_recon)
                        )
                    total_recon_loss += recon_loss.item()

                total_loss += loss_val
                n_batches += 1

            avg_loss = total_loss / max(n_batches, 1)
            avg_recon_loss = total_recon_loss / max(n_batches, 1)

            if use_wandb:
                import wandb
                wandb.log({
                    f"pretrain/layer{layer_idx + 1}/free_energy_diff": avg_loss,
                    f"pretrain/layer{layer_idx + 1}/recon_loss": avg_recon_loss,
                    "pretrain/epoch": epoch,
                    "pretrain/layer": layer_idx + 1,
                })

            if verbose >= 1 and epoch % 10 == 0:
                loss_name = "Recon MSE (mean)" if use_gbrbm_for_this_layer else "Recon BCE (mean)"
                print(f"  Epoch {epoch:3d}: Free Energy Diff = {avg_loss:.4f}, {loss_name} = {avg_recon_loss:.4f}")

        if use_wandb:
            import wandb
            wandb.run.summary[f"pretrain/layer{layer_idx + 1}/final_energy_diff"] = avg_loss
            wandb.run.summary[f"pretrain/layer{layer_idx + 1}/final_recon_loss"] = avg_recon_loss

        rbm_list.append(rbm)

        # 次の層のための入力データを生成（現在のRBMの隠れ層表現）
        if layer_idx < n_layers - 1:
            print(f"  📊 Generating latent representations for next layer...")
            with torch.no_grad():
                rbm.eval()
                next_data = []
                for i in range(0, len(current_data), pcd_batch_size):
                    batch_data = current_data[i:i + pcd_batch_size].to(device)
                    _, h_prob = rbm.sample_h_given_v(batch_data)

                    # 【重要】確率値ではなく、サンプリングして0/1にする
                    # Binary RBMは v ∈ {0,1} を想定しているため、
                    # 次の層への入力も二値化する必要がある。
                    # (Hinton et al. 2006, Deep Belief Networks)
                    h_sample = torch.bernoulli(h_prob)

                    next_data.append(h_sample)
                current_data = torch.cat(next_data, dim=0).cpu()
            print(f"     → Sampled {len(current_data)} binary vectors for next layer")

    print(f"\n{'='*60}")
    print(f"✅ Greedy Pretraining Completed!")
    print(f"{'='*60}")

    # DBM構築と初期化
    dbm = DeepBoltzmannMachine(layer_sizes, use_gbrbm=use_gbrbm,
                               sigma_init=sigma_init, learn_sigma=learn_sigma,
                               lag_window=lag_window, J_comp_per_layer=J_comp_per_layer,
                               mf_init=mf_init)
    dbm.initialize_from_rbms(rbm_list, weight_scaling=weight_scaling)
    dbm.to(device)
    if compensate_biases:
        dbm.compensate_biases(original_data, device)
    else:
        print("   (Bias compensation disabled)")

    return dbm, rbm_list

def test_dbm(batch_size, layer_sizes, device):
    print("=" * 60)
    print("Testing DeepBoltzmannMachine...")
    print("=" * 60)

    # テスト用のDBM構築（3層: 可視層20, 隠れ層1: 16, 隠れ層2: 8）
    dbm = DeepBoltzmannMachine(layer_sizes).to(device)

    print(f"Architecture: {layer_sizes}")
    print(f"Number of hidden layers: {dbm.n_layers}")
    print()

    # テストデータ（二値入力）
    v_test = (torch.randn(batch_size, layer_sizes[0]) > 0).float().to(device)

    # Mean-field inference
    print("Testing mean_field_inference...")
    h_probs = dbm.mean_field_inference(v_test, n_iter=10)
    for i, h in enumerate(h_probs):
        print(f"  h{i+1}_prob shape: {h.shape}, range: [{h.min():.3f}, {h.max():.3f}]")

    # サンプリング
    print()
    print("Testing sample_h_given_v...")
    h_samples, h_probs = dbm.sample_h_given_v(v_test, n_iter=10)
    for i, (h_s, h_p) in enumerate(zip(h_samples, h_probs)):
        print(f"  h{i+1}_sample shape: {h_s.shape}, unique: {torch.unique(h_s).tolist()}")

    # 可視層の再構成
    print()
    print("Testing sample_v_given_h...")
    v_recon, v_prob = dbm.sample_v_given_h(h_samples)
    print(f"  v_recon shape: {v_recon.shape}")
    print(f"  v_prob range: [{v_prob.min():.3f}, {v_prob.max():.3f}]")

    # Gibbs sampling
    print()
    print("Testing gibbs_sampling...")
    v_gibbs, h_gibbs = dbm.gibbs_sampling(v_test, k_steps=3, n_iter=5)
    print(f"  v_gibbs shape: {v_gibbs.shape}")
    print(f"  h_gibbs layers: {len(h_gibbs)}")

    # Free energy
    print()
    print("Testing free_energy...")
    fe = dbm.free_energy(v_test, n_iter=10)
    print(f"  Free energy shape: {fe.shape}, mean: {fe.mean():.3f}")

    # Latent representation
    print()
    print("Testing get_latent_representation...")
    latent_top = dbm.get_latent_representation(v_test, use_top_layer=True)
    latent_all = dbm.get_latent_representation(v_test, use_top_layer=False)
    print(f"  Top layer only: {latent_top.shape}")
    print(f"  All layers concatenated: {latent_all.shape}")

    # RBMからの初期化テスト
    print()
    print("=" * 60)
    print("Testing initialize_from_rbms...")
    print("=" * 60)

    # 個別のRBMを作成（layer_sizesに基づいて動的に生成）
    rbm_list = []
    for i in range(len(layer_sizes) - 1):
        rbm = RestrictedBoltzmannMachine(layer_sizes[i], layer_sizes[i + 1]).to(device)
        rbm_list.append(rbm)

    # 新しいDBMを作成して初期化
    dbm2 = DeepBoltzmannMachine(layer_sizes).to(device)
    dbm2.initialize_from_rbms(rbm_list)

    # 初期化後の推論テスト
    h_probs2 = dbm2.mean_field_inference(v_test, n_iter=10)
    for i, h in enumerate(h_probs2):
        print(f"  After initialization, h{i+1}_prob mean: {h.mean():.3f}")

def test_gbdbm(batch_size, layer_sizes, device):
    print()
    print("=" * 60)
    print("Testing DeepBoltzmannMachine with GBRBM mode...")
    print("=" * 60)

    n_visible = layer_sizes[0]

    # GBRBM モードのDBM構築
    dbm_gbrbm = DeepBoltzmannMachine(layer_sizes, use_gbrbm=True, sigma_init=1.0, learn_sigma=False).to(device)

    print(f"Architecture: {layer_sizes} (GBRBM mode)")
    print(f"use_gbrbm: {dbm_gbrbm.use_gbrbm}")
    print(f"sigma_sq shape: {dbm_gbrbm.sigma_sq.shape}, mean: {dbm_gbrbm.sigma_sq.mean():.3f}")
    print()

    # テストデータ（連続値入力）
    v_test_cont = torch.randn(batch_size, n_visible).to(device)

    # Mean-field inference
    print("Testing mean_field_inference (GBRBM)...")
    h_probs_gbrbm = dbm_gbrbm.mean_field_inference(v_test_cont, n_iter=10)
    for i, h in enumerate(h_probs_gbrbm):
        print(f"  h{i+1}_prob shape: {h.shape}, range: [{h.min():.3f}, {h.max():.3f}]")

    # サンプリング
    print()
    print("Testing sample_h_given_v (GBRBM)...")
    h_samples_gbrbm, h_probs_gbrbm = dbm_gbrbm.sample_h_given_v(v_test_cont, n_iter=10)
    for i, (h_s, h_p) in enumerate(zip(h_samples_gbrbm, h_probs_gbrbm)):
        print(f"  h{i+1}_sample shape: {h_s.shape}, unique: {torch.unique(h_s).tolist()}")

    # 可視層の再構成（ガウス分布）
    print()
    print("Testing sample_v_given_h (GBRBM, Gaussian output)...")
    v_recon_gbrbm, v_mean_gbrbm = dbm_gbrbm.sample_v_given_h(h_samples_gbrbm, add_noise=True)
    print(f"  v_recon shape: {v_recon_gbrbm.shape}")
    print(f"  v_recon range: [{v_recon_gbrbm.min():.3f}, {v_recon_gbrbm.max():.3f}]")
    print(f"  v_mean range: [{v_mean_gbrbm.min():.3f}, {v_mean_gbrbm.max():.3f}]")

    # ノイズなしの再構成
    v_recon_no_noise, _ = dbm_gbrbm.sample_v_given_h(h_samples_gbrbm, add_noise=False)
    print(f"  v_recon (no noise) range: [{v_recon_no_noise.min():.3f}, {v_recon_no_noise.max():.3f}]")

    # Gibbs sampling
    print()
    print("Testing gibbs_sampling (GBRBM)...")
    v_gibbs_gbrbm, h_gibbs_gbrbm = dbm_gbrbm.gibbs_sampling(v_test_cont, k_steps=3, n_iter=5)
    print(f"  v_gibbs shape: {v_gibbs_gbrbm.shape}")
    print(f"  v_gibbs range: [{v_gibbs_gbrbm.min():.3f}, {v_gibbs_gbrbm.max():.3f}]")

    # Free energy
    print()
    print("Testing free_energy (GBRBM)...")
    fe_gbrbm = dbm_gbrbm.free_energy(v_test_cont, n_iter=10)
    print(f"  Free energy shape: {fe_gbrbm.shape}, mean: {fe_gbrbm.mean():.3f}")

    # Latent representation
    print()
    print("Testing get_latent_representation (GBRBM)...")
    latent_top_gbrbm = dbm_gbrbm.get_latent_representation(v_test_cont, use_top_layer=True)
    latent_all_gbrbm = dbm_gbrbm.get_latent_representation(v_test_cont, use_top_layer=False)
    print(f"  Top layer only: {latent_top_gbrbm.shape}")
    print(f"  All layers concatenated: {latent_all_gbrbm.shape}")

    # GBRBM + BB-RBM からの初期化テスト
    print()
    print("=" * 60)
    print("Testing initialize_from_rbms with GBRBM...")
    print("=" * 60)

    # 個別のRBMを作成（最初の層はGBRBM、残りはBB-RBM）
    rbm_list_gbrbm = []
    for i in range(len(layer_sizes) - 1):
        if i == 0:
            rbm = GaussianBernoulliRBM(layer_sizes[i], layer_sizes[i + 1], learn_sigma=False, sigma_init=1.0).to(device)
        else:
            rbm = RestrictedBoltzmannMachine(layer_sizes[i], layer_sizes[i + 1]).to(device)
        rbm_list_gbrbm.append(rbm)

    # 新しいDBMを作成して初期化
    dbm_gbrbm2 = DeepBoltzmannMachine(layer_sizes, use_gbrbm=True, sigma_init=0.5, learn_sigma=False).to(device)
    print(f"  Before init, sigma mean: {dbm_gbrbm2.sigma.mean():.3f}")
    dbm_gbrbm2.initialize_from_rbms(rbm_list_gbrbm)
    print(f"  After init, sigma mean: {dbm_gbrbm2.sigma.mean():.3f}")

    # 初期化後の推論テスト
    h_probs_gbrbm2 = dbm_gbrbm2.mean_field_inference(v_test_cont, n_iter=10)
    for i, h in enumerate(h_probs_gbrbm2):
        print(f"  After initialization, h{i+1}_prob mean: {h.mean():.3f}")

    # learn_sigma=True のテスト
    print()
    print("=" * 60)
    print("Testing GBRBM with learnable sigma...")
    print("=" * 60)

    dbm_learn_sigma = DeepBoltzmannMachine(layer_sizes, use_gbrbm=True, sigma_init=1.0, learn_sigma=True).to(device)
    print(f"  learn_sigma: {dbm_learn_sigma.learn_sigma}")
    print(f"  log_sigma shape: {dbm_learn_sigma.log_sigma.shape}")
    print(f"  sigma_sq (from log): {dbm_learn_sigma.sigma_sq.mean():.3f}")

    # 推論テスト
    h_probs_learn = dbm_learn_sigma.mean_field_inference(v_test_cont, n_iter=10)
    print(f"  h1_prob mean: {h_probs_learn[0].mean():.3f}")

def test_lrdbm(batch_size, layer_sizes, device):
    """LR-DBM (Latent-Recurrent DBM) のテスト"""
    from collections import deque

    print()
    print("=" * 60)
    print("Testing LR-DBM (Latent-Recurrent DBM)...")
    print("=" * 60)

    # LR-DBM パラメータ
    lag_window = 3
    d_v_original = layer_sizes[0]  # 元の可視層次元
    hidden_dims = list(layer_sizes[1:])  # 隠れ層のサイズ（ラグとは独立）
    n_hidden_layers = len(hidden_dims)

    # 各隠れ層を1/4に圧縮（最小1）
    J_comp_per_layer = [max(1, dim // 4) for dim in hidden_dims]
    d_comp = sum(J_comp_per_layer)

    # 拡張入力次元: d_v + lag_window * d_comp（参考用、内部で自動計算される）
    expected_extended_dim = d_v_original + lag_window * d_comp

    # LR-DBM用のlayer_sizes: 元の可視層次元 + 隠れ層
    # 注意: 新設計では layer_sizes[0] は「元の観測次元」を指定する
    lrdbm_layer_sizes = [d_v_original] + hidden_dims

    print(f"  layer_sizes (user specified): {lrdbm_layer_sizes}")
    print(f"  d_v_original: {d_v_original}")
    print(f"  hidden_dims: {hidden_dims}")
    print(f"  lag_window: {lag_window}")
    print(f"  J_comp_per_layer: {J_comp_per_layer}")
    print(f"  d_comp: {d_comp}")
    print(f"  expected_extended_dim: {expected_extended_dim}")
    print()

    # LR-DBM構築
    lrdbm = DeepBoltzmannMachine(
        lrdbm_layer_sizes,
        use_gbrbm=True,  # 連続値入力
        lag_window=lag_window,
        J_comp_per_layer=J_comp_per_layer
    ).to(device)

    print(f"LR-DBM Architecture:")
    print(f"  layer_sizes (specified): {lrdbm.layer_sizes}")
    print(f"  d_v_original: {lrdbm.d_v_original}")
    print(f"  d_v_extended (auto-calculated): {lrdbm.d_v_extended}")
    print(f"  lag_window: {lrdbm.lag_window}")
    print(f"  J_comp_per_layer: {lrdbm.J_comp_per_layer}")
    print(f"  d_comp: {lrdbm.d_comp}")
    for i in range(n_hidden_layers):
        print(f"  W[{i}] shape: {lrdbm.weights[i].shape}")
    print()

    # 拡張入力次元が正しく計算されているか確認
    assert lrdbm.d_v_extended == expected_extended_dim, \
        f"Expected d_v_extended={expected_extended_dim}, got {lrdbm.d_v_extended}"
    assert lrdbm.weights[0].shape == (expected_extended_dim, hidden_dims[0]), \
        f"Expected W[0] shape=({expected_extended_dim}, {hidden_dims[0]}), got {lrdbm.weights[0].shape}"

    # compress() テスト
    print("Testing compress()...")
    dummy_h_layers = [torch.randn(batch_size, dim).to(device) for dim in hidden_dims]
    compressed = lrdbm.compress(dummy_h_layers)
    print(f"  Input h_layers shapes: {[h.shape for h in dummy_h_layers]}")
    print(f"  Compressed shape: {compressed.shape}")
    assert compressed.shape == (batch_size, d_comp), f"Expected ({batch_size}, {d_comp}), got {compressed.shape}"

    # prepare_history() テスト
    print()
    print("Testing prepare_history()...")
    history_queue = deque(maxlen=lag_window)

    # 履歴が空の場合（t=0）
    history_list = lrdbm.prepare_history(history_queue, device=device)
    print(f"  t=0 (empty queue): len(history_list)={len(history_list)}")
    assert len(history_list) == lag_window, f"Expected {lag_window}, got {len(history_list)}"

    # 履歴を追加
    for t in range(lag_window + 2):
        c_t = torch.randn(batch_size, d_comp).to(device)
        history_queue.append(c_t)
        history_list = lrdbm.prepare_history(history_queue, device=device)
        print(f"  t={t+1}: queue_len={len(history_queue)}, history_list_len={len(history_list)}")

    # build_extended_input() テスト
    print()
    print("Testing build_extended_input()...")
    v_t = torch.randn(batch_size, d_v_original).to(device)
    history_list = lrdbm.prepare_history(history_queue, device=device)
    v_bar = lrdbm.build_extended_input(v_t, history_list)
    print(f"  v_t shape: {v_t.shape}")
    print(f"  v_bar shape: {v_bar.shape}")  # 期待: (batch, 38)
    assert v_bar.shape == (batch_size, expected_extended_dim), \
        f"Expected ({batch_size}, {expected_extended_dim}), got {v_bar.shape}"

    # Mean-field inference テスト（拡張入力）
    print()
    print("Testing mean_field_inference with extended input...")
    h_probs = lrdbm.mean_field_inference(v_bar, n_iter=10)
    for i, h in enumerate(h_probs):
        print(f"  h{i+1}_prob shape: {h.shape}, range: [{h.min():.3f}, {h.max():.3f}]")
        # 隠れ層のサイズが正しいか確認
        assert h.shape == (batch_size, hidden_dims[i]), \
            f"Expected h{i+1} shape=({batch_size}, {hidden_dims[i]}), got {h.shape}"

    # 時系列シミュレーション
    print()
    print("Simulating time series processing...")
    history_queue = deque(maxlen=lag_window)
    T = 10  # 時系列長

    for t in range(T):
        # 観測データ
        v_t = torch.randn(batch_size, d_v_original).to(device)

        # 履歴準備
        history_list = lrdbm.prepare_history(history_queue, device=device)

        # 拡張入力構築
        v_bar = lrdbm.build_extended_input(v_t, history_list)

        # 推論
        h_probs = lrdbm.mean_field_inference(v_bar, n_iter=5)

        # 圧縮して履歴に追加
        compressed = lrdbm.compress(h_probs)
        history_queue.append(compressed.detach())

        if t < 5 or t == T - 1:
            print(f"  t={t}: v_bar shape={v_bar.shape}, compressed shape={compressed.shape}")

    print()
    print("✅ LR-DBM tests passed!")

def test(device, batch_size = 32):
    test_dbm(batch_size=batch_size, layer_sizes = [24, 8], device=device)
    test_dbm(batch_size=batch_size, layer_sizes = [36, 18, 4], device=device)
    test_dbm(batch_size=batch_size, layer_sizes = [32, 16, 8, 2], device=device)
    #test_gbdbm(batch_size=batch_size,layer_sizes=layer_sizes, device=device)
    #test_lrdbm(batch_size=batch_size,layer_sizes=layer_sizes, device=device)

    print()
    print("=" * 60)
    print("All DBM tests passed!")
    print("=" * 60)

if __name__ == '__main__':
    # デバイスの設定
    device = find_device()
    print(f"Using device: {device}")

    test(device=device)
