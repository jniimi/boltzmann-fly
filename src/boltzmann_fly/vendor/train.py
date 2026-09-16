import os, pickle, random, gc
try:
    import wandb
except ImportError:  # boltzmann-fly: wandb is optional (only used when use_wandb=True)
    wandb = None
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Union
from collections import deque

from tqdm.auto import tqdm
import pandas as pd
try:
    import matplotlib.pyplot as plt
except ImportError:  # boltzmann-fly: matplotlib is optional
    plt = None

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from .dbm import DeepBoltzmannMachine, greedy_pretrain_dbm
from .rbm import RestrictedBoltzmannMachine, GaussianBernoulliRBM
from .utils import check_config, find_device, send_message

# ===========================
# Phase 1: Greedy Layer-wise Pretraining
# ===========================
def pretrain_bm(
    config: Dict[str, Any],
    dataloader: DataLoader,
    use_wandb: bool = False,
    verbose: int = 2,
) -> Tuple[DeepBoltzmannMachine, List[Any]]:
    """
    Greedy Layer-wise Pretrainingを実行してDBMを事前学習する。

    Args:
        config: 設定辞書
        dataloader: 学習データのDataLoader
        use_wandb: wandbでログを記録するかどうか

    Returns:
        Tuple[DeepBoltzmannMachine, List[Any]]: 事前学習済みDBMと各層のRBMリスト
    """
    print("\n" + "="*70)
    print("PHASE 1: GREEDY LAYER-WISE PRETRAINING")
    print("="*70)

    dbm, rbm_list = greedy_pretrain_dbm(
        layer_sizes=config["bm"].get("layer_sizes"),
        dataloader=dataloader,
        device=config["device"],
        n_epochs_per_layer=config["bm"].get("nepochs_greedy_pretraining"),
        lr=config["bm"]["pretraining_lr"],
        k_steps=config["bm"]["pretraining_ksteps"],
        pcd_batch_size=config["bm"].get("batchsize"),
        use_gbrbm=config["bm"].get("use_gbrbm", False),
        sigma_init=config["bm"].get("sigma_init", 1.0),
        learn_sigma=config["bm"].get("learn_sigma", False),
        weight_init=config["bm"].get("weight_init", "xavier_normal"),
        weight_scaling=config["bm"].get("weight_scaling", None),
        mf_init=config["bm"].get("mf_init", 0.5),
        pretraining_weight_decay=config["bm"].get("pretraining_weight_decay", 1e-3),
        compensate_biases=config["bm"].get("compensate_biases", True),
        use_wandb=use_wandb,
        verbose=verbose,
    )

    # Save pretrained DBM
    f_dbm_pretrain = config["drive_dir"] / "models" / f"{config['bm']['save_pretrain_id']}.pkl"
    with f_dbm_pretrain.open("wb") as fp:
        pickle.dump(dbm, fp)
    print(f"💾 Pretrained DBM saved to {f_dbm_pretrain}")

    message = "DBM pretrained"
    if config["verbose"].get("send_message", False):
        send_message(message, web_hook_url=config["verbose"]["slack_webhook_url"])
    return dbm, rbm_list

# ===========================
# Phase 2: Joint Fine-tuning
# ===========================
def joint_finetuning(
    config: Dict[str, Any],
    dbm: DeepBoltzmannMachine,
    dataloader: DataLoader,
    val_dataloader: DataLoader = None,
    use_wandb: bool = False,
    verbose: int = 2,
) -> DeepBoltzmannMachine:
    """
    事前学習済みDBMのJoint Fine-tuningを実行する。

    val_dataloaderが与えられた場合、validation recon BCEに基づく
    early stoppingを行い、bestモデルを復元する。

    Args:
        config: 設定辞書
        dbm: 事前学習済みのDeepBoltzmannMachine
        dataloader: 学習データのDataLoader
        val_dataloader: バリデーション用DataLoader（Noneでearly stopping無効）
        use_wandb: wandbでログを記録するかどうか

    Returns:
        DeepBoltzmannMachine: Fine-tuning済みのDBM
    """
    print("\n" + "="*70)
    print("PHASE 2: JOINT FINE-TUNING")
    print("="*70)

    device = config["device"]
    batchsize = config["bm"].get("batchsize")
    n_visible = config["dataset"]["n_visible"]
    n_epochs_joint = config["bm"].get("nepochs_joint_finetuning")

    lr = config["bm"]["finetuning_lr"]
    decay = config["bm"]["finetuning_decay"]

    ksteps = config["bm"]["finetuning_ksteps"]
    niter = config["bm"]["finetuning_niter"]
    grad_clip = config["bm"].get("finetuning_grad_clip", None)
    patience = config["bm"].get("finetuning_patience", 20)

    if n_epochs_joint > 0:
        print(f"🔧 Fine-tuning DBM for {n_epochs_joint} epochs...")
        if grad_clip is not None:
            print(f"   Gradient clipping: max_norm={grad_clip}")
        if val_dataloader is not None:
            print(f"   Early stopping: patience={patience} (val recon BCE)")

        dbm.to(device)
        optimizer_dbm = torch.optim.Adam(dbm.parameters(), lr=lr, weight_decay=decay)

        if dbm.use_gbrbm:
            pcd_buffer = torch.randn(batchsize, n_visible).to(device)
        else:
            pcd_buffer = torch.bernoulli(torch.rand(batchsize, n_visible)).to(device)

        # 診断用データを収集（全データ）
        diag_data_list = []
        for batch in dataloader:
            diag_data_list.append(batch['gbm_vector'])
        diag_data = torch.cat(diag_data_list, dim=0)

        # 診断間隔
        diag_interval = max(1, n_epochs_joint // 5) if n_epochs_joint > 5 else 1

        # FT開始前の診断
        if verbose >= 1:
            diagnose_activations(dbm, diag_data, device, label="FT Epoch 0 (before)")

        # Early stopping state
        best_val_recon = float('inf')
        best_state = None
        epochs_no_improve = 0

        dbm.train()
        for epoch in tqdm(range(n_epochs_joint), desc="DBM Joint Training", disable=(verbose < 2)):
            total_loss = 0
            total_recon_loss = 0

            for batch in dataloader:
                v_pos = batch['gbm_vector'].to(device)

                loss_val, v_neg_new = dbm.train_step(
                    v_pos, optimizer_dbm, pcd_buffer,
                    k_steps=ksteps, n_iter=niter,
                    grad_clip=grad_clip,
                )

                pcd_buffer = v_neg_new.detach()
                total_loss += loss_val

                with torch.no_grad():
                    h_probs = dbm.mean_field_inference(v_pos, n_iter=niter)
                    v_recon, _ = dbm.sample_v_given_h(h_probs, add_noise=False)

                    if dbm.use_gbrbm:
                        recon_loss = torch.mean((v_pos - v_recon) ** 2)
                    else:
                        eps = 1e-7
                        v_recon = torch.clamp(v_recon, eps, 1.0 - eps)
                        recon_loss = -torch.mean(
                            v_pos * torch.log(v_recon) +
                            (1 - v_pos) * torch.log(1 - v_recon)
                        )
                    total_recon_loss += recon_loss.item()

            avg_loss = total_loss / len(dataloader)
            avg_recon_loss = total_recon_loss / len(dataloader)

            # Validation early stopping
            val_recon_str = ""
            if val_dataloader is not None:
                val_metrics = evaluate_dbm(dbm, val_dataloader, device, n_iter=niter)
                val_recon = val_metrics["recon_bce"]
                improved = val_recon < best_val_recon
                if improved:
                    best_val_recon = val_recon
                    best_state = {k: v.cpu().clone() for k, v in dbm.state_dict().items()}
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                val_recon_str = f", Val BCE = {val_recon:.4f}{'*' if improved else ''}"
                dbm.train()  # evaluate_dbm sets eval mode

            log_dict = {
                "finetune/free_energy_diff": avg_loss,
                "finetune/recon_loss": avg_recon_loss,
                "finetune/epoch": epoch,
            }
            if val_dataloader is not None:
                log_dict["finetune/val_recon_bce"] = val_recon
            if use_wandb:
                wandb.log(log_dict)

            if verbose >= 1 and (epoch % 10 == 0 or (val_dataloader is not None and epochs_no_improve == 0)):
                loss_name = "Recon MSE (mean)" if dbm.use_gbrbm else "Recon BCE (mean)"
                print(f"  Epoch {epoch}: Free Energy Diff = {avg_loss:.4f}, "
                      f"{loss_name} = {avg_recon_loss:.4f}{val_recon_str}")

            # エポック別診断
            if verbose >= 1 and ((epoch + 1) % diag_interval == 0 or epoch == n_epochs_joint - 1):
                diagnose_activations(dbm, diag_data, device, label=f"FT Epoch {epoch + 1}")

            # Early stopping check
            if val_dataloader is not None and epochs_no_improve >= patience:
                print(f"  Early stopping at epoch {epoch} (best val recon BCE = {best_val_recon:.4f})")
                break

        # Restore best model if early stopping was used
        if best_state is not None:
            dbm.load_state_dict(best_state)
            dbm.to(device)
            print(f"  Restored best model (val recon BCE = {best_val_recon:.4f})")

        # 最終epochのenergy diffをsummaryに記録
        if use_wandb:
            wandb.run.summary["finetune/final_energy_diff"] = avg_loss
            wandb.run.summary["finetune/final_recon_loss"] = avg_recon_loss
            if val_dataloader is not None:
                wandb.run.summary["finetune/best_val_recon_bce"] = best_val_recon

        print("✅ DBM Joint Fine-tuning Completed.")

        # Save fine-tuned DBM
        f_dbm_finetuned = config["drive_dir"] / "models" / f"{config['bm']['save_finetuning_id']}.pkl"
        with f_dbm_finetuned.open("wb") as fp:
            pickle.dump(dbm, fp)
        print(f"💾 Fine-tuned DBM saved to {f_dbm_finetuned}")
    else:
        print("⏭️  Skipping joint fine-tuning (n_epochs_joint=0)")

    message = "DBM finetuned"
    if config["verbose"].get("send_message", False):
        send_message(message, web_hook_url=config["verbose"]["slack_webhook_url"])

    return dbm


# ===========================
# Evaluation
# ===========================
def evaluate_dbm(
    dbm: DeepBoltzmannMachine,
    dataloader: DataLoader,
    device: torch.device,
    n_iter: int = 10,
) -> Dict[str, float]:
    """
    DBMの評価: Free EnergyとReconstruction BCE を計算する。

    Args:
        dbm: 評価対象のDBM
        dataloader: 評価データのDataLoader
        device: torch.device
        n_iter: Mean-field反復回数

    Returns:
        {"free_energy_abs": float, "recon_bce": float}
    """
    dbm.eval()
    dbm.to(device)

    total_energy = 0.0
    total_recon = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            v = batch['gbm_vector'].to(device)

            energy = dbm.free_energy(v, n_iter=n_iter).mean().item()

            h_probs = dbm.mean_field_inference(v, n_iter=n_iter)
            v_recon, _ = dbm.sample_v_given_h(h_probs, add_noise=False)
            eps = 1e-7
            v_recon = torch.clamp(v_recon, eps, 1.0 - eps)
            recon = -torch.mean(
                v * torch.log(v_recon) + (1 - v) * torch.log(1 - v_recon)
            ).item()

            total_energy += energy
            total_recon += recon
            n_batches += 1

    return {
        "free_energy_abs": total_energy / max(n_batches, 1),
        "recon_bce": total_recon / max(n_batches, 1),
    }


# ===========================
# Diagnosis
# ===========================
def diagnose_activations(
    dbm: DeepBoltzmannMachine,
    data: torch.Tensor,
    device: torch.device,
    n_iter: int = 10,
    batch_size: int = 256,
    label: str = "Pretrained",
) -> Dict[str, Any]:
    """
    DBMの各隠れ層の活性化状態を診断する。

    Pretraining直後やFine-tuning後に呼び出して、
    dead units / saturated units の発生状況を確認する。

    Args:
        dbm: 診断対象のDBM
        data: 入力データ (N, n_visible) or (N, d_v_extended) for LR-DBM
        device: 計算デバイス
        n_iter: Mean-field反復回数
        batch_size: バッチサイズ
        label: 表示用ラベル

    Returns:
        Dict: 各層の活性化統計
    """
    dbm.eval()
    dbm.to(device)

    all_h_probs = [[] for _ in range(dbm.n_layers)]

    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size].to(device)
            h_probs = dbm.mean_field_inference(batch, n_iter=n_iter)
            for layer_idx, h in enumerate(h_probs):
                all_h_probs[layer_idx].append(h.cpu())

    print(f"\n{'='*60}")
    print(f"ACTIVATION DIAGNOSIS ({label})")
    print(f"{'='*60}")

    results = {}
    for layer_idx in range(dbm.n_layers):
        h_all = torch.cat(all_h_probs[layer_idx], dim=0)
        n_samples, n_units = h_all.shape
        mean_act = h_all.mean(dim=0)
        std_act = h_all.std(dim=0)

        dead = (mean_act < 0.01).sum().item()
        saturated = (mean_act > 0.99).sum().item()
        low_var = (std_act < 0.01).sum().item()

        print(f"\n📊 Layer {layer_idx + 1} ({n_units} units, {n_samples} samples):")
        print(f"   Mean activation: {mean_act.mean():.4f} ± {mean_act.std():.4f}")
        print(f"   Dead (<0.01): {dead} ({100 * dead / n_units:.1f}%)")
        print(f"   Saturated (>0.99): {saturated} ({100 * saturated / n_units:.1f}%)")
        print(f"   Low variance: {low_var} ({100 * low_var / n_units:.1f}%)")

        results[f"layer_{layer_idx + 1}"] = {
            "mean_activation": mean_act.mean().item(),
            "std_activation": mean_act.std().item(),
            "dead_units": dead,
            "saturated_units": saturated,
            "low_var_units": low_var,
            "n_units": n_units,
        }

    return results


def diagnose_rbm_activations(
    rbm: nn.Module,
    data: torch.Tensor,
    device: torch.device,
    batch_size: int = 256,
    label: str = "",
) -> Dict[str, Any]:
    """
    RBM単体の隠れ層活性化を診断する（pretraining中のエポック別監視用）。

    Args:
        rbm: GBRBM or BB-RBM
        data: 入力データ (N, n_visible)
        device: 計算デバイス
        batch_size: バッチサイズ
        label: 表示用ラベル（例: "Epoch 0", "Init"）

    Returns:
        Dict: 活性化統計
    """
    rbm.eval()
    all_h_probs = []

    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size].to(device)
            _, h_prob = rbm.sample_h_given_v(batch)
            all_h_probs.append(h_prob.cpu())

    h_all = torch.cat(all_h_probs, dim=0)
    n_samples, n_units = h_all.shape
    mean_act = h_all.mean(dim=0)
    std_act = h_all.std(dim=0)

    dead = (mean_act < 0.01).sum().item()
    saturated = (mean_act > 0.99).sum().item()
    low_var = (std_act < 0.01).sum().item()

    print(f"    [{label}] mean={mean_act.mean():.4f}±{mean_act.std():.4f}, "
          f"dead={dead}/{n_units}({100*dead/n_units:.0f}%), "
          f"sat={saturated}/{n_units}({100*saturated/n_units:.0f}%), "
          f"lowvar={low_var}/{n_units}({100*low_var/n_units:.0f}%)")

    rbm.train()
    return {
        "mean_activation": mean_act.mean().item(),
        "std_activation": mean_act.std().item(),
        "dead_units": dead,
        "saturated_units": saturated,
        "low_var_units": low_var,
        "n_units": n_units,
    }


# ===========================
# LR-DBM (Latent-Recurrent DBM)
# ===========================
def pretrain_lrdbm(
    config: Dict[str, Any],
    dataset,  # get_grouped_by_consumer() を持つデータセット
    use_wandb: bool = False,
    verbose: int = 2,
) -> Tuple[DeepBoltzmannMachine, List[Any]]:
    """
    LR-DBM用のGreedy Layer-wise Pretraining

    第1層は拡張入力（v_t + 圧縮履歴のゼロパディング）を受け取る。
    Pretraining時は履歴がないためゼロで代用する。

    Args:
        config: 設定辞書
        dataset: get_grouped_by_consumer()メソッドを持つデータセット
        use_wandb: wandbでログを記録するかどうか

    Returns:
        Tuple[DeepBoltzmannMachine, List[Any]]: LR-DBMとRBMリスト
    """
    print("\n" + "="*70)
    print("PHASE 1: LR-DBM GREEDY LAYER-WISE PRETRAINING")
    print("="*70)

    device = config["device"]
    layer_sizes = config["bm"]["layer_sizes"]
    n_epochs = config["bm"].get("nepochs_greedy_pretraining")
    lr = config["bm"]["pretraining_lr"]
    k_steps = config["bm"]["pretraining_ksteps"]
    pcd_batch_size = config["bm"].get("batchsize")
    use_gbrbm = config["bm"].get("use_gbrbm", True)
    sigma_init = config["bm"].get("sigma_init", 1.0)
    learn_sigma = config["bm"].get("learn_sigma", False)

    lag_window = config["bm"].get("lag_window", 3)
    J_comp_per_layer = config["bm"].get("J_comp_per_layer", None)

    n_layers = len(layer_sizes) - 1
    n_visible_original = layer_sizes[0]
    hidden_dims = layer_sizes[1:]

    # 圧縮次元の計算
    if J_comp_per_layer is None:
        J_comp_per_layer = [max(1, dim // 4) for dim in hidden_dims]
    d_comp = sum(J_comp_per_layer)

    # 拡張入力次元
    n_visible_extended = n_visible_original + lag_window * d_comp

    print(f"  Original visible dim: {n_visible_original}")
    print(f"  Extended visible dim: {n_visible_extended}")
    print(f"  Lag window: {lag_window}, d_comp: {d_comp}")

    # 消費者ごとのデータから元の観測ベクトルのみ収集（履歴なし）
    # Pretraining時は履歴次元を含めず、元の可視層次元のみでRBMを学習する。
    # 履歴部分の重みはDBM構築時にXavier初期化し、FTで学習する。
    consumer_data = dataset.get_grouped_by_consumer()
    all_vectors = []

    for consumer_id in sorted(consumer_data.keys()):
        vectors = consumer_data[consumer_id]  # (T, n_visible_original)
        all_vectors.append(vectors)

    current_data = torch.cat(all_vectors, dim=0)  # (N_total, n_visible_original)
    original_data = current_data.clone()  # 診断用に元データを保持（層ループで上書きされるため）
    print(f"  Total samples: {current_data.shape[0]}, shape: {current_data.shape}")
    print(f"  ℹ️  Pretraining on original dims only; history weights will be Xavier-initialized")

    # 各層を順次RBMとして学習
    rbm_list = []
    for layer_idx in range(n_layers):
        if layer_idx == 0:
            n_visible = n_visible_original  # 履歴なしの元次元でRBMを学習
        else:
            n_visible = layer_sizes[layer_idx]
        n_hidden = layer_sizes[layer_idx + 1]
        use_gbrbm_for_this_layer = (layer_idx == 0 and use_gbrbm)

        layer_type = "GBRBM" if use_gbrbm_for_this_layer else "RBM"
        print(f"\n  Layer {layer_idx + 1}/{n_layers}: {layer_type}({n_visible} -> {n_hidden})")

        if use_gbrbm_for_this_layer:
            rbm = GaussianBernoulliRBM(n_visible, n_hidden, learn_sigma=learn_sigma, sigma_init=sigma_init,
                                       weight_init=config["bm"].get("weight_init", "xavier_normal")).to(device)
            v_mean = current_data.mean(dim=0).to(device)
            rbm.v_bias.data = v_mean
        else:
            rbm = RestrictedBoltzmannMachine(n_visible, n_hidden).to(device)
            if layer_idx == 0:
                v_mean = current_data.mean(dim=0).to(device)
                eps = 1e-4
                v_mean = torch.clamp(v_mean, eps, 1.0 - eps)
                rbm.v_bias.data = torch.log(v_mean / (1.0 - v_mean))

        pretraining_weight_decay = config["bm"].get("pretraining_weight_decay", 1e-3)
        optimizer = torch.optim.Adam(rbm.parameters(), lr=lr, weight_decay=pretraining_weight_decay)

        if use_gbrbm_for_this_layer:
            pcd_buffer = torch.randn(pcd_batch_size, n_visible).to(device)
        else:
            pcd_buffer = torch.bernoulli(torch.rand(pcd_batch_size, n_visible)).to(device)

        # 診断間隔: 10エポックごと（最初と最後は必ず実行）
        diag_interval = max(1, n_epochs // 10) if n_epochs > 10 else 1

        # 初期化直後の診断
        if verbose >= 1:
            diagnose_rbm_activations(rbm, current_data, device, label=f"L{layer_idx+1} Init")

        rbm.train()
        for epoch in tqdm(range(n_epochs), desc=f"{layer_type} Layer {layer_idx + 1}", disable=(verbose < 2)):
            total_loss = 0
            n_batches = 0
            indices = torch.randperm(len(current_data))

            for i in range(0, len(current_data), pcd_batch_size):
                batch_indices = indices[i:i + pcd_batch_size]
                if len(batch_indices) < pcd_batch_size:
                    continue
                batch_data = current_data[batch_indices].to(device)
                loss_val, v_neg = rbm.train_step(batch_data, optimizer, pcd_buffer, k_steps)
                pcd_buffer = v_neg.detach()
                total_loss += loss_val
                n_batches += 1

            avg_loss = total_loss / max(n_batches, 1)

            if use_wandb:
                wandb.log({
                    f"pretrain_lrdbm/layer{layer_idx + 1}/free_energy_diff": avg_loss,
                    "pretrain_lrdbm/epoch": epoch,
                    "pretrain_lrdbm/layer": layer_idx + 1,
                })

            if verbose >= 1 and epoch % 10 == 0:
                print(f"    Epoch {epoch}: Free Energy Diff = {avg_loss:.4f}")

            # エポック別活性化診断
            if verbose >= 1 and (epoch % diag_interval == 0 or epoch == n_epochs - 1):
                diagnose_rbm_activations(rbm, current_data, device, label=f"L{layer_idx+1} Ep{epoch}")

        rbm_list.append(rbm)

        # 次の層のための入力データ生成
        if layer_idx < n_layers - 1:
            with torch.no_grad():
                rbm.eval()
                next_data = []
                for i in range(0, len(current_data), pcd_batch_size):
                    batch_data = current_data[i:i + pcd_batch_size].to(device)
                    _, h_prob = rbm.sample_h_given_v(batch_data)
                    h_sample = torch.bernoulli(h_prob)
                    next_data.append(h_sample.cpu())
                current_data = torch.cat(next_data, dim=0)

    # LR-DBM構築と初期化
    mf_init = config["bm"].get("mf_init", 0.5)
    dbm = DeepBoltzmannMachine(
        layer_sizes, use_gbrbm=use_gbrbm,
        sigma_init=sigma_init, learn_sigma=learn_sigma,
        lag_window=lag_window, J_comp_per_layer=J_comp_per_layer,
        mf_init=mf_init,
    )
    weight_scaling = config["bm"].get("weight_scaling", None)
    dbm.initialize_from_rbms(rbm_list, weight_scaling=weight_scaling)
    dbm.to(device)

    # LR-DBMの場合、ゼロ履歴で拡張入力を構築
    zeros_history = torch.zeros(len(original_data), lag_window * d_comp)
    diag_data = torch.cat([original_data, zeros_history], dim=1)

    if config["bm"].get("compensate_biases", True):
        dbm.compensate_biases(diag_data, device)
    else:
        print("   (Bias compensation disabled)")
    diagnose_activations(dbm, diag_data, device, label="After Pretraining (LR-DBM)")

    # Save
    f_dbm_pretrain = config["drive_dir"] / "models" / f"{config['bm']['save_pretrain_id']}.pkl"
    with f_dbm_pretrain.open("wb") as fp:
        pickle.dump(dbm, fp)
    print(f"💾 Pretrained LR-DBM saved to {f_dbm_pretrain}")

    return dbm, rbm_list


def joint_finetuning_lrdbm(
    config: Dict[str, Any],
    dbm: DeepBoltzmannMachine,
    dataset,  # get_grouped_by_consumer() を持つデータセット
    use_wandb: bool = False,
    verbose: int = 2,
) -> DeepBoltzmannMachine:
    """
    LR-DBM用のJoint Fine-tuning（消費者バッチ並列版）

    時系列は逐次処理（履歴依存のため）だが、同一時点の消費者は
    独立なので全消費者を1バッチとして並列処理する。
    ループ回数: epochs × T（従来の epochs × N_consumers × T から大幅削減）

    Args:
        config: 設定辞書
        dbm: 事前学習済みLR-DBM
        dataset: get_grouped_by_consumer()メソッドを持つデータセット
        use_wandb: wandbでログを記録するかどうか

    Returns:
        DeepBoltzmannMachine: Fine-tuning済みのLR-DBM
    """
    print("\n" + "="*70)
    print("PHASE 2: LR-DBM JOINT FINE-TUNING")
    print("="*70)

    device = config["device"]
    n_epochs = config["bm"].get("nepochs_joint_finetuning")
    lr = config["bm"]["finetuning_lr"]
    decay = config["bm"]["finetuning_decay"]
    ksteps = config["bm"]["finetuning_ksteps"]
    niter = config["bm"]["finetuning_niter"]
    grad_clip = config["bm"].get("finetuning_grad_clip", None)

    lag_window = dbm.lag_window

    if n_epochs <= 0:
        print("⏭️  Skipping LR-DBM joint fine-tuning (n_epochs=0)")
        return dbm

    dbm.to(device)
    optimizer = torch.optim.Adam(dbm.parameters(), lr=lr, weight_decay=decay)

    # 消費者データを (n_consumers, T, n_visible) に整形
    consumer_data = dataset.get_grouped_by_consumer()
    consumer_ids = sorted(consumer_data.keys())
    n_consumers = len(consumer_ids)
    T = consumer_data[consumer_ids[0]].shape[0]
    all_vectors = torch.stack([consumer_data[cid] for cid in consumer_ids]).to(device)  # (N, T, d_v)

    print(f"🔧 Fine-tuning LR-DBM for {n_epochs} epochs...")
    print(f"   {n_consumers} consumers × {T} periods, batch per step = {n_consumers}")
    print(f"   use_wandb={use_wandb}, wandb.run={'active ('+wandb.run.id+')' if wandb.run else 'None'}")
    if grad_clip is not None:
        print(f"   Gradient clipping: max_norm={grad_clip}")

    # 診断用データ（ゼロ履歴の拡張入力）
    d_v_original = all_vectors.shape[2]
    diag_data_flat = all_vectors.reshape(-1, d_v_original).cpu()  # (N*T, d_v)
    zeros_history = torch.zeros(diag_data_flat.shape[0], lag_window * dbm.d_comp)
    diag_data = torch.cat([diag_data_flat, zeros_history], dim=1)

    # 診断間隔
    diag_interval = max(1, n_epochs // 5) if n_epochs > 5 else 1

    # FT開始前の診断
    if verbose >= 1:
        diagnose_activations(dbm, diag_data, device, label="FT Epoch 0 (before)")

    # PCDバッファ: 消費者数分（拡張入力次元）
    n_visible_extended = dbm.d_v_extended
    if dbm.use_gbrbm:
        pcd_buffer = torch.randn(n_consumers, n_visible_extended).to(device)
    else:
        pcd_buffer = torch.bernoulli(torch.rand(n_consumers, n_visible_extended)).to(device)

    dbm.train()
    for epoch in tqdm(range(n_epochs), desc="LR-DBM Joint Training", disable=(verbose < 2)):
        total_loss = 0.0
        total_recon = 0.0

        # エポック開始時に全消費者の履歴をリセット
        # history_buffer[i] = i番目に新しい圧縮表現 (n_consumers, d_comp)
        # [0]=c(t-1), [1]=c(t-2), ..., [s-1]=c(t-s)
        history_buffer = [
            torch.zeros(n_consumers, dbm.d_comp, device=device)
            for _ in range(lag_window)
        ]

        for t in range(T):
            v_t = all_vectors[:, t, :]  # (n_consumers, n_visible)

            # 拡張入力を構築（build_extended_inputはバッチ対応済み）
            v_bar = dbm.build_extended_input(v_t, history_buffer)

            # 全消費者を1バッチとして学習
            loss_val, v_neg_new = dbm.train_step(
                v_bar, optimizer, pcd_buffer,
                k_steps=ksteps, n_iter=niter,
                grad_clip=grad_clip,
            )
            pcd_buffer = v_neg_new.detach()
            total_loss += loss_val

            # 圧縮表現を計算して履歴バッファを更新
            with torch.no_grad():
                h_probs = dbm.mean_field_inference(v_bar, n_iter=niter)
                compressed = dbm.compress(h_probs)  # (n_consumers, d_comp)

                # 履歴シフト: 最古を捨てて最新を先頭に挿入
                history_buffer = [compressed.detach()] + history_buffer[:-1]

                # Reconstruction Loss
                v_recon, _ = dbm.sample_v_given_h(h_probs, add_noise=False)
                if dbm.use_gbrbm:
                    recon = torch.mean((v_bar - v_recon) ** 2).item()
                else:
                    eps = 1e-7
                    v_recon = torch.clamp(v_recon, eps, 1.0 - eps)
                    recon = -torch.mean(
                        v_bar * torch.log(v_recon) + (1 - v_bar) * torch.log(1 - v_recon)
                    ).item()
                total_recon += recon

        avg_loss = total_loss / T
        avg_recon = total_recon / T

        if use_wandb:
            wandb.log({
                "finetune_lrdbm/free_energy_diff": avg_loss,
                "finetune_lrdbm/recon_loss": avg_recon,
                "finetune_lrdbm/epoch": epoch,
            })

        if verbose >= 1:
            loss_name = "Recon MSE (mean)" if dbm.use_gbrbm else "Recon BCE (mean)"
            print(f"  Epoch {epoch}: Free Energy Diff = {avg_loss:.4f}, {loss_name} = {avg_recon:.4f}")

        # エポック別診断
        if verbose >= 1 and ((epoch + 1) % diag_interval == 0 or epoch == n_epochs - 1):
            diagnose_activations(dbm, diag_data, device, label=f"FT Epoch {epoch + 1}")

    # 最終epochのenergy diffをsummaryに記録
    if use_wandb:
        wandb.run.summary["finetune_lrdbm/final_energy_diff"] = avg_loss
        wandb.run.summary["finetune_lrdbm/final_recon_loss"] = avg_recon

    print("✅ LR-DBM Joint Fine-tuning Completed.")

    # Save
    f_dbm_finetuned = config["drive_dir"] / "models" / f"{config['bm']['save_finetuning_id']}.pkl"
    with f_dbm_finetuned.open("wb") as fp:
        pickle.dump(dbm, fp)
    print(f"💾 Fine-tuned LR-DBM saved to {f_dbm_finetuned}")

    return dbm

class MockDataset(Dataset):
    """テスト用のモックデータセット"""
    def __init__(self, n_samples: int, n_visible: int, continuous: bool = False):
        """
        Args:
            n_samples: サンプル数
            n_visible: 可視層の次元数
            continuous: Trueの場合は連続値、Falseの場合は二値データ
        """
        self.n_samples = n_samples
        self.n_visible = n_visible
        self.continuous = continuous

        if continuous:
            # 連続値データ（GBRBM用）
            self.data = torch.randn(n_samples, n_visible)
        else:
            # 二値データ（BB-RBM用）
            self.data = (torch.randn(n_samples, n_visible) > 0).float()

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return {'gbm_vector': self.data[idx]}


class MockTimeSeriesDataset:
    """LR-DBMテスト用のモック時系列データセット（消費者×期間構造）"""
    def __init__(self, n_consumers: int, n_periods: int, n_visible: int, continuous: bool = True):
        self.n_consumers = n_consumers
        self.n_periods = n_periods
        self.n_visible = n_visible
        self.continuous = continuous

        # 消費者ごと・期間ごとのデータを生成
        if continuous:
            self.data = torch.randn(n_consumers * n_periods, n_visible)
        else:
            self.data = (torch.randn(n_consumers * n_periods, n_visible) > 0).float()

        # consumer_id と period の対応を保持
        self.consumer_ids = []
        self.periods = []
        for cid in range(n_consumers):
            for t in range(n_periods):
                self.consumer_ids.append(cid)
                self.periods.append(t)

    def get_grouped_by_consumer(self) -> Dict[int, torch.Tensor]:
        """消費者ごとにグループ化したデータを返す（LR-DBM用）"""
        result = {}
        idx = 0
        for cid in range(self.n_consumers):
            vectors = self.data[idx:idx + self.n_periods]
            result[cid] = vectors
            idx += self.n_periods
        return result

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {'gbm_vector': self.data[idx]}


def create_test_config(
    drive_dir: Path,
    n_visible: int = 20,
    layer_sizes: List[int] = None,
    use_gbrbm: bool = False,
    lag_window: int = 0,
    J_comp_per_layer: List[int] = None,
) -> Dict[str, Any]:
    """
    テスト用の設定辞書を作成する。

    Args:
        drive_dir: 保存先ディレクトリ
        n_visible: 可視層の次元数
        layer_sizes: 各層のサイズ（Noneの場合は自動設定）
        use_gbrbm: GBRBMモードを使用するかどうか
        lag_window: LR-DBMのラグ窓幅（0の場合は通常DBM）
        J_comp_per_layer: 各隠れ層の圧縮次元数

    Returns:
        設定辞書
    """
    if layer_sizes is None:
        layer_sizes = [n_visible, 16, 8]

    config = {
        "device": find_device(),
        "drive_dir": drive_dir,
        "dataset": {
            "n_visible": n_visible,
        },
        "bm": {
            "layer_sizes": layer_sizes,
            "batchsize": 32,
            # Greedy pretraining
            "nepochs_greedy_pretraining": 5,  # テスト用に少なく
            "pretraining_lr": 0.01,
            "pretraining_ksteps": 1,
            "save_pretrain_id": "test_dbm_pretrain",
            # Joint fine-tuning
            "nepochs_joint_finetuning": 3,  # テスト用に少なく
            "finetuning_lr": 0.001,
            "finetuning_decay": 1e-4,
            "finetuning_ksteps": 1,
            "finetuning_niter": 5,
            "save_finetuning_id": "test_dbm_finetuned",
            # GBRBM設定
            "use_gbrbm": use_gbrbm,
            "sigma_init": 1.0,
            "learn_sigma": False,
            # LR-DBM設定
            "lag_window": lag_window,
            "J_comp_per_layer": J_comp_per_layer,
        },
        "gpt": {},
        "adapter": {},
        "verbose": {
            "send_message": False,
        },
    }
    return config


def test_pretrain_bm(device, tmp_dir: Path):
    """pretrain_bm関数のテスト"""
    print("=" * 60)
    print("Testing pretrain_bm (BB-RBM mode)...")
    print("=" * 60)

    # モックデータ作成
    n_samples = 256
    n_visible = 20
    dataset = MockDataset(n_samples, n_visible, continuous=False)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 設定作成
    config = create_test_config(tmp_dir, n_visible=n_visible)

    # 事前学習実行
    dbm, rbm_list = pretrain_bm(config, dataloader)

    # 検証
    assert dbm is not None, "DBM should be created"
    assert len(rbm_list) == len(config["bm"]["layer_sizes"]) - 1, "RBM list length mismatch"
    assert dbm.n_layers == 2, f"Expected 2 hidden layers, got {dbm.n_layers}"

    # 推論テスト
    dbm.to(device)
    v_test = (torch.randn(8, n_visible) > 0).float().to(device)
    h_probs = dbm.mean_field_inference(v_test, n_iter=5)
    assert len(h_probs) == 2, "Should have 2 hidden layers"

    # 保存ファイルの確認
    save_path = tmp_dir / "models" / f"{config['bm']['save_pretrain_id']}.pkl"
    assert save_path.exists(), f"Pretrained model should be saved at {save_path}"

    print("✅ pretrain_bm test passed!")
    return dbm


def test_joint_finetuning(device, dbm: DeepBoltzmannMachine, tmp_dir: Path):
    """joint_finetuning関数のテスト"""
    print()
    print("=" * 60)
    print("Testing joint_finetuning...")
    print("=" * 60)

    # モックデータ作成
    n_samples = 256
    n_visible = 20
    dataset = MockDataset(n_samples, n_visible, continuous=False)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 設定作成
    config = create_test_config(tmp_dir, n_visible=n_visible)

    # Fine-tuning実行
    dbm_finetuned = joint_finetuning(config, dbm, dataloader)

    # 検証
    assert dbm_finetuned is not None, "Fine-tuned DBM should be returned"

    # 推論テスト
    dbm_finetuned.to(device)
    v_test = (torch.randn(8, n_visible) > 0).float().to(device)
    h_probs = dbm_finetuned.mean_field_inference(v_test, n_iter=5)
    assert len(h_probs) == 2, "Should have 2 hidden layers"

    # 保存ファイルの確認
    save_path = tmp_dir / "models" / f"{config['bm']['save_finetuning_id']}.pkl"
    assert save_path.exists(), f"Fine-tuned model should be saved at {save_path}"

    print("✅ joint_finetuning test passed!")
    return dbm_finetuned


def test_gbrbm_mode(device, tmp_dir: Path):
    """GBRBMモードでのテスト"""
    print()
    print("=" * 60)
    print("Testing pretrain_bm with GBRBM mode...")
    print("=" * 60)

    # モックデータ作成（連続値）
    n_samples = 256
    n_visible = 20
    dataset = MockDataset(n_samples, n_visible, continuous=True)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 設定作成（GBRBMモード）
    config = create_test_config(tmp_dir, n_visible=n_visible, use_gbrbm=True)
    config["bm"]["save_pretrain_id"] = "test_gbdbm_pretrain"
    config["bm"]["save_finetuning_id"] = "test_gbdbm_finetuned"

    # 事前学習実行
    dbm, rbm_list = pretrain_bm(config, dataloader)

    # 検証
    assert dbm is not None, "DBM should be created"
    assert dbm.use_gbrbm, "DBM should be in GBRBM mode"
    assert len(rbm_list) == 2, "Should have 2 RBMs"

    # 第1層がGBRBMであることを確認
    from rbm import GaussianBernoulliRBM
    assert isinstance(rbm_list[0], GaussianBernoulliRBM), "First RBM should be GBRBM"

    # 推論テスト（連続値入力）
    dbm.to(device)
    v_test = torch.randn(8, n_visible).to(device)
    h_probs = dbm.mean_field_inference(v_test, n_iter=5)
    assert len(h_probs) == 2, "Should have 2 hidden layers"

    # 可視層再構成テスト（ガウス出力）
    h_samples = [torch.bernoulli(h) for h in h_probs]
    v_recon, _ = dbm.sample_v_given_h(h_samples, add_noise=False)
    assert v_recon.shape == v_test.shape, "Reconstructed shape should match input"

    print("✅ GBRBM mode pretrain test passed!")

    # GBRBM Fine-tuningテスト
    print()
    print("=" * 60)
    print("Testing joint_finetuning with GBRBM mode...")
    print("=" * 60)

    dbm_finetuned = joint_finetuning(config, dbm, dataloader)

    assert dbm_finetuned.use_gbrbm, "Fine-tuned DBM should still be in GBRBM mode"

    print("✅ GBRBM mode fine-tuning test passed!")
    return dbm_finetuned


def test_lrdbm_pretrain(device, tmp_dir: Path):
    """LR-DBM pretrain_lrdbm関数のテスト"""
    print()
    print("=" * 60)
    print("Testing pretrain_lrdbm (LR-DBM GBRBM mode)...")
    print("=" * 60)

    import time

    n_consumers = 8
    n_periods = 10
    n_visible = 20
    lag_window = 3

    dataset = MockTimeSeriesDataset(n_consumers, n_periods, n_visible, continuous=True)

    config = create_test_config(
        tmp_dir, n_visible=n_visible, use_gbrbm=True,
        lag_window=lag_window,
    )
    config["bm"]["save_pretrain_id"] = "test_lrdbm_pretrain"
    config["bm"]["save_finetuning_id"] = "test_lrdbm_finetuned"

    start = time.time()
    dbm, rbm_list = pretrain_lrdbm(config, dataset)
    elapsed = time.time() - start
    print(f"  ⏱️  Pretrain elapsed: {elapsed:.2f}s")

    # 検証
    assert dbm is not None, "LR-DBM should be created"
    assert dbm.lag_window == lag_window, f"Expected lag_window={lag_window}, got {dbm.lag_window}"
    assert dbm.d_v_extended > n_visible, "Extended dim should be larger than original"
    assert len(rbm_list) == 2, "Should have 2 RBMs"

    # 推論テスト（拡張入力）
    dbm.to(device)
    v_bar_test = torch.randn(4, dbm.d_v_extended).to(device)
    h_probs = dbm.mean_field_inference(v_bar_test, n_iter=5)
    assert len(h_probs) == 2, "Should have 2 hidden layers"

    # prepare_history / build_extended_input テスト
    history_queue = deque(maxlen=lag_window)
    v_t = torch.randn(1, n_visible).to(device)
    history_list = dbm.prepare_history(history_queue, device=device)
    v_bar = dbm.build_extended_input(v_t, history_list)
    assert v_bar.shape == (1, dbm.d_v_extended), f"Expected shape (1, {dbm.d_v_extended}), got {v_bar.shape}"

    # 保存ファイルの確認
    save_path = tmp_dir / "models" / "test_lrdbm_pretrain.pkl"
    assert save_path.exists(), f"Pretrained LR-DBM should be saved at {save_path}"

    print("✅ pretrain_lrdbm test passed!")
    return dbm


def test_lrdbm_finetuning(device, dbm: DeepBoltzmannMachine, tmp_dir: Path):
    """LR-DBM joint_finetuning_lrdbm関数のテスト"""
    print()
    print("=" * 60)
    print("Testing joint_finetuning_lrdbm...")
    print("=" * 60)

    import time

    n_consumers = 8
    n_periods = 10
    n_visible = 20
    lag_window = dbm.lag_window

    dataset = MockTimeSeriesDataset(n_consumers, n_periods, n_visible, continuous=True)

    config = create_test_config(
        tmp_dir, n_visible=n_visible, use_gbrbm=True,
        lag_window=lag_window,
    )
    config["bm"]["save_pretrain_id"] = "test_lrdbm_pretrain"
    config["bm"]["save_finetuning_id"] = "test_lrdbm_finetuned"
    config["bm"]["nepochs_joint_finetuning"] = 2  # テスト用に少なく

    start = time.time()
    dbm_ft = joint_finetuning_lrdbm(config, dbm, dataset)
    elapsed = time.time() - start

    n_total_samples = n_consumers * n_periods
    n_epochs = config["bm"]["nepochs_joint_finetuning"]
    print(f"  ⏱️  FT elapsed: {elapsed:.2f}s ({n_total_samples} samples × {n_epochs} epochs)")
    print(f"  ⏱️  Per sample-epoch: {elapsed / (n_total_samples * n_epochs) * 1000:.1f}ms")

    # 検証
    assert dbm_ft is not None, "Fine-tuned LR-DBM should be returned"
    assert dbm_ft.lag_window == lag_window, "lag_window should be preserved"

    # 推論テスト（時系列的にhistoryを使った推論ができるか）
    dbm_ft.to(device)
    dbm_ft.eval()
    history_queue = deque(maxlen=lag_window)

    for t in range(5):
        v_t = torch.randn(1, n_visible).to(device)
        history_list = dbm_ft.prepare_history(history_queue, device=device)
        v_bar = dbm_ft.build_extended_input(v_t, history_list)
        assert v_bar.shape == (1, dbm_ft.d_v_extended)

        with torch.no_grad():
            h_probs = dbm_ft.mean_field_inference(v_bar, n_iter=5)
            compressed = dbm_ft.compress(h_probs)
            history_queue.append(compressed.detach())

    assert len(history_queue) == min(5, lag_window), "History queue should be filled"

    # 保存ファイルの確認
    save_path = tmp_dir / "models" / "test_lrdbm_finetuned.pkl"
    assert save_path.exists(), f"Fine-tuned LR-DBM should be saved at {save_path}"

    print("✅ joint_finetuning_lrdbm test passed!")
    return dbm_ft


def test(device):
    """全テストを実行"""
    import tempfile
    import shutil

    # 一時ディレクトリ作成
    tmp_dir = Path(tempfile.mkdtemp())
    (tmp_dir / "models").mkdir(parents=True, exist_ok=True)

    try:
        # BB-RBMモードのテスト
        dbm = test_pretrain_bm(device, tmp_dir)
        test_joint_finetuning(device, dbm, tmp_dir)

        # GBRBMモードのテスト
        test_gbrbm_mode(device, tmp_dir)

        # LR-DBMモードのテスト
        lrdbm = test_lrdbm_pretrain(device, tmp_dir)
        test_lrdbm_finetuning(device, lrdbm, tmp_dir)

        print()
        print("=" * 60)
        print("All train.py tests passed!")
        print("=" * 60)

    finally:
        # 一時ディレクトリ削除
        shutil.rmtree(tmp_dir)
        print(f"\n🧹 Cleaned up temporary directory: {tmp_dir}")


if __name__ == "__main__":
    device = find_device()
    print(f"Using device: {device}")

    test(device=device)
