"""
Adapter Module: Prediction Heads on Frozen DBM Beliefs

DBM（World Model = "脳"）の凍結済みbelief表現にAdapter（"口"）を接続し、
visit/purchase の予測モデルを構築する。Phase 3 に相当。

Author: Junichiro Niimi
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Any
from collections import deque

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score,
)
from scipy.stats import spearmanr

from .dbm import DeepBoltzmannMachine
from .utils import find_device


# ============================================================
# 1. PredictionAdapter
# ============================================================
class PredictionAdapter(nn.Module):
    """
    [belief, actions] → MLP → Linear(1)

    forward() は logit を返す（Sigmoidは損失関数側で制御）。
    beliefs: DBMの潜在表現、actions: 当期の施策変数 Z_t

    hidden_dim: int または List[int] で隠れ層構造を指定。
        int: 1層MLP (後方互換)
        List[int]: 多層MLP (例: [64, 32, 16])
    """

    def __init__(self, belief_dim: int, action_dim: int = 0,
                 hidden_dim=64,
                 dropout: float = 0.1, task_name: str = ""):
        super().__init__()
        self.task_name = task_name
        self.belief_dim = belief_dim
        self.action_dim = action_dim

        if isinstance(hidden_dim, int):
            hidden_dims = [hidden_dim]
        else:
            hidden_dims = list(hidden_dim)

        input_dim = belief_dim + action_dim
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, belief: torch.Tensor,
                actions: torch.Tensor = None) -> torch.Tensor:
        """Returns logit (batch,)"""
        if actions is not None:
            x = torch.cat([belief, actions], dim=-1)
        else:
            x = belief
        return self.net(x).squeeze(-1)


# ============================================================
# 2. AdapterDataset
# ============================================================
class AdapterDataset(Dataset):
    """事前抽出済みbelief・アクション変数とターゲットをペアにするDataset。"""

    def __init__(self, beliefs: torch.Tensor, actions: torch.Tensor,
                 visit: torch.Tensor, purchase: torch.Tensor,
                 metadata: pd.DataFrame = None):
        """
        Args:
            beliefs: (N, belief_dim)
            actions: (N, action_dim) — 当期の施策変数 Z_t
            visit: (N,)
            purchase: (N,)
            metadata: 行順対応のDataFrame（consumer_id, day, true_* 等）
        """
        self.beliefs = beliefs
        self.actions = actions
        self.visit = visit
        self.purchase = purchase
        self.metadata = metadata

    def __len__(self):
        return len(self.beliefs)

    def __getitem__(self, idx):
        return {
            'belief': self.beliefs[idx],
            'actions': self.actions[idx],
            'visit': self.visit[idx],
            'purchase': self.purchase[idx],
        }


# ============================================================
# 3. extract_beliefs — Standard DBM用
# ============================================================
def extract_beliefs(
    dbm: DeepBoltzmannMachine,
    dataset,  # SimulationDataset
    device: torch.device,
    use_top_layer: bool = True,
    n_iter: int = 10,
    batch_size: int = 256,
    forward: int = 0,
) -> AdapterDataset:
    """
    Standard DBM からbelief表現を抽出し AdapterDataset として返す。

    1. DBMを凍結（eval + requires_grad=False）
    2. DataLoaderで全データを処理
    3. get_latent_representation で belief 抽出
    4. dataset.get_actions() で当期施策変数 Z_t を取得
    5. dataset.get_targets() / get_metadata() でラベル・メタデータ取得

    forward > 0 の場合、t期のbelief/actionsとt+forward期のターゲットをペアにする。
    t+forward期が存在しない行はスキップされる。
    """
    dbm.eval()
    for p in dbm.parameters():
        p.requires_grad = False

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_beliefs = []

    with torch.no_grad():
        for batch in loader:
            v = batch['gbm_vector'].to(device)
            belief = dbm.get_latent_representation(v, n_iter=n_iter,
                                                    use_top_layer=use_top_layer)
            all_beliefs.append(belief.cpu())

    beliefs = torch.cat(all_beliefs, dim=0)
    actions = dataset.get_actions()
    targets = dataset.get_targets()
    metadata = dataset.get_metadata()

    if forward == 0:
        return AdapterDataset(beliefs, actions, targets['visit'], targets['purchase'], metadata)

    # forward > 0: t期のbelief/actionsとt+forward期のターゲットをペアにする
    df = dataset.df.reset_index(drop=True)
    lookup = {}
    for idx, row in df.iterrows():
        lookup[(int(row['consumer_id']), int(row['day']))] = idx

    keep_belief_indices = []
    keep_target_indices = []
    for i in range(len(df)):
        cid = int(df.iloc[i]['consumer_id'])
        t = int(df.iloc[i]['day'])
        target_key = (cid, t + forward)
        if target_key in lookup:
            keep_belief_indices.append(i)
            keep_target_indices.append(lookup[target_key])

    beliefs_paired = beliefs[keep_belief_indices]
    actions_paired = actions[keep_belief_indices]
    visit_paired = targets['visit'][keep_target_indices]
    purchase_paired = targets['purchase'][keep_target_indices]
    meta_paired = metadata.iloc[keep_target_indices].reset_index(drop=True)

    return AdapterDataset(beliefs_paired, actions_paired, visit_paired, purchase_paired, meta_paired)


# ============================================================
# 4. extract_beliefs_lrdbm — LR-DBM用
# ============================================================
def extract_beliefs_lrdbm(
    dbm: DeepBoltzmannMachine,
    dataset,  # SimulationDataset
    device: torch.device,
    use_top_layer: bool = True,
    n_iter: int = 10,
    forward: int = 0,
) -> AdapterDataset:
    """
    LR-DBM からbelief表現を抽出し AdapterDataset として返す。

    消費者別に時系列を再生し、history_queue を管理しながら
    各時点の belief を抽出する。

    forward > 0 の場合、t期のbeliefとt+forward期のターゲットをペアにする。
    同消費者内でt+forward期が存在しない行はスキップされる。
    """
    dbm.eval()
    for p in dbm.parameters():
        p.requires_grad = False

    consumer_data = dataset.get_grouped_by_consumer()
    lag_window = dbm.lag_window

    # consumer_id, period の対応表を構築
    df = dataset.df.reset_index(drop=True)
    sorted_df = df.sort_values(['consumer_id', 'period']).reset_index(drop=True)

    # forward > 0 用: (consumer_id, period) → sorted_df行インデックス のルックアップ辞書
    if forward > 0:
        target_lookup = {}
        for idx, row in sorted_df.iterrows():
            target_lookup[(int(row['consumer_id']), int(row['period']))] = idx

    all_beliefs = []
    ordered_indices = []  # 元のdfでの行順を記録（ターゲット行）

    with torch.no_grad():
        for cid in sorted(consumer_data.keys()):
            vectors = consumer_data[cid]  # (T, n_visible)
            T = vectors.shape[0]
            history_queue = deque(maxlen=lag_window)

            # cid の行をperiod順に取得
            cid_rows = sorted_df[sorted_df['consumer_id'] == cid]

            for t in range(T):
                v_t = vectors[t:t + 1].to(device)
                history_list = dbm.prepare_history(history_queue, device=device)
                v_bar = dbm.build_extended_input(v_t, history_list)

                h_probs = dbm.mean_field_inference(v_bar, n_iter=n_iter)
                if use_top_layer:
                    belief = h_probs[-1]
                else:
                    belief = torch.cat(h_probs, dim=1)

                # 履歴更新
                compressed = dbm.compress(h_probs)
                history_queue.append(compressed.detach())

                if forward == 0:
                    all_beliefs.append(belief.cpu())
                    ordered_indices.append(cid_rows.index[t])
                else:
                    # t+forward期のターゲットを検索
                    period_t = int(cid_rows.iloc[t]['period'])
                    target_key = (cid, period_t + forward)
                    if target_key in target_lookup:
                        all_beliefs.append(belief.cpu())
                        ordered_indices.append(target_lookup[target_key])

    beliefs = torch.cat(all_beliefs, dim=0)

    # ターゲットとメタデータを抽出順に再整列
    targets = dataset.get_targets()
    metadata = dataset.get_metadata()

    visit = targets['visit'][ordered_indices]
    purchase = targets['purchase'][ordered_indices]
    meta_ordered = metadata.iloc[ordered_indices].reset_index(drop=True)

    return AdapterDataset(beliefs, visit, purchase, meta_ordered)


# ============================================================
# 5. train_adapter
# ============================================================
def train_adapter(
    adapter: PredictionAdapter,
    train_dataset: AdapterDataset,
    val_dataset: AdapterDataset,
    task: str,
    n_epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 256,
    device: torch.device = None,
    patience: int = 10,
    use_wandb: bool = False,
    wandb_prefix: str = "",
    use_pos_weight: bool = False,
    verbose: int = 2,
) -> Tuple[PredictionAdapter, Dict[str, float]]:
    """
    Adapterの学習。

    - Visit: 全サンプルで BCEWithLogitsLoss
    - Purchase: visit=1 のサンプルのみで BCEWithLogitsLoss
    - Early stopping: val AUC が patience エポック改善なしで停止
    - ReduceLROnPlateau スケジューラ
    """
    if device is None:
        device = find_device()

    adapter.to(device)
    adapter.train()

    pw = None
    if use_pos_weight:
        if task == 'visit':
            targets_all = train_dataset.visit
        else:
            targets_all = train_dataset.purchase
        n_pos = targets_all.sum().item()
        n_neg = len(targets_all) - n_pos
        if n_pos > 0 and n_neg > 0:
            pw = torch.tensor([n_neg / n_pos], device=device)
            print(f"  [{adapter.task_name}] pos_weight={pw.item():.3f} "
                  f"(pos={int(n_pos)}, neg={int(n_neg)})")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    optimizer = torch.optim.Adam(adapter.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    best_val_auc = -1.0
    best_state = None
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_auc': []}

    for epoch in range(n_epochs):
        # --- Train ---
        adapter.train()
        total_loss = 0.0
        n_samples = 0

        for batch in train_loader:
            beliefs = batch['belief'].to(device)
            actions = batch['actions'].to(device)
            target = batch[task].to(device)

            logits = adapter(beliefs, actions)
            loss = criterion(logits, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(target)
            n_samples += len(target)

        avg_loss = total_loss / max(n_samples, 1)
        history['train_loss'].append(avg_loss)

        # --- Evaluate ---
        train_metrics = evaluate_adapter(adapter, train_dataset, task, device)
        train_auc = train_metrics.get('auc', 0.0)
        val_metrics = evaluate_adapter(adapter, val_dataset, task, device)
        val_auc = val_metrics.get('auc', 0.0)
        history['val_auc'].append(val_auc)

        scheduler.step(val_auc)

        # wandb logging
        if use_wandb:
            import wandb
            log_dict = {
                f"{wandb_prefix}train_bce": avg_loss,
                f"{wandb_prefix}train_auc": train_auc,
                f"{wandb_prefix}val_auc": val_auc,
                f"{wandb_prefix}val_loss": val_metrics['loss'],
                f"{wandb_prefix}val_accuracy": val_metrics['accuracy'],
                f"{wandb_prefix}epoch": epoch,
            }
            wandb.log(log_dict)

        # Early stopping
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in adapter.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if verbose >= 1 and (epoch % 10 == 0 or epochs_no_improve == 0):
            print(f"  [{adapter.task_name}] Epoch {epoch}: "
                  f"train_bce={avg_loss:.4f}, train_auc={train_auc:.4f}, val_auc={val_auc:.4f}"
                  f"{' *' if epochs_no_improve == 0 else ''}")

        if epochs_no_improve >= patience:
            if verbose >= 0:
                print(f"  [{adapter.task_name}] Early stopping at epoch {epoch} "
                      f"(best val_auc={best_val_auc:.4f})")
            break

    # Restore best model
    if best_state is not None:
        adapter.load_state_dict(best_state)
    adapter.to(device)

    return adapter, {'best_val_auc': best_val_auc, **history}


# ============================================================
# 6. evaluate_adapter
# ============================================================
def evaluate_adapter(
    adapter: PredictionAdapter,
    dataset: AdapterDataset,
    task: str,
    device: torch.device,
    batch_size: int = 512,
    threshold: float = None,
) -> Dict[str, float]:
    """
    Adapterの評価。

    返却: {loss, accuracy, auc, precision, recall, f1, best_threshold}
    Purchase は visit=1 サブセットで評価。

    Args:
        threshold: 指定時はそのthresholdで分類指標を算出（探索しない）。
                   Noneの場合はF1最大化で自動探索。
    """
    adapter.eval()
    criterion = nn.BCEWithLogitsLoss()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_logits = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            beliefs = batch['belief'].to(device)
            actions = batch['actions'].to(device)
            target = batch[task]

            logits = adapter(beliefs, actions)
            all_logits.append(logits.cpu())
            all_targets.append(target)

    if len(all_logits) == 0:
        return {'loss': float('nan'), 'accuracy': 0.0, 'auc': 0.0,
                'precision': 0.0, 'recall': 0.0, 'f1': 0.0,
                'best_threshold': 0.5}

    logits = torch.cat(all_logits)
    targets = torch.cat(all_targets)

    loss = criterion(logits, targets).item()
    probs = torch.sigmoid(logits).numpy()
    y_true = targets.numpy()

    # AUC: 片方のクラスしかない場合は 0.0
    try:
        auc = roc_auc_score(y_true, probs)
    except ValueError:
        auc = 0.0

    # Threshold: 指定があればそのまま使用、なければF1最大化で探索
    if threshold is not None:
        best_thresh = threshold
    else:
        best_f1 = 0.0
        best_thresh = 0.5
        for th in np.arange(0.1, 0.9, 0.05):
            preds_th = (probs > th).astype(float)
            f1_th = f1_score(y_true, preds_th, zero_division=0)
            if f1_th > best_f1:
                best_f1 = f1_th
                best_thresh = th

    preds = (probs > best_thresh).astype(float)
    acc = accuracy_score(y_true, preds)
    prec = precision_score(y_true, preds, zero_division=0)
    rec = recall_score(y_true, preds, zero_division=0)
    f1 = f1_score(y_true, preds, zero_division=0)

    return {'loss': loss, 'accuracy': acc, 'auc': auc,
            'precision': prec, 'recall': rec, 'f1': f1,
            'best_threshold': best_thresh}


# ============================================================
# 7. predict_intervention_effects
# ============================================================
def predict_intervention_effects(
    dbm: DeepBoltzmannMachine,
    adapter: PredictionAdapter,
    adapter_dataset: AdapterDataset,
    action_columns: List[str],
    intervention_col: str,
    baseline_val: float,
    treatment_val: float,
    device: torch.device,
    batch_size: int = 256,
) -> pd.DataFrame:
    """
    反事実予測: actions内の intervention_col を baseline/treatment に設定した時の
    予測差（uplift）を計算する。

    Z_tはDBMではなくadapterの入力なので、beliefは固定のまま
    actions側のみ変更して予測を比較する。
    """
    adapter.eval()

    # intervention_col の actions テンソル内でのインデックス
    action_idx = action_columns.index(intervention_col)

    loader = DataLoader(adapter_dataset, batch_size=batch_size, shuffle=False)

    all_logit_base = []
    all_logit_treat = []

    with torch.no_grad():
        for batch in loader:
            beliefs = batch['belief'].to(device)
            actions = batch['actions'].to(device)

            # Baseline
            a_base = actions.clone()
            a_base[:, action_idx] = baseline_val
            all_logit_base.append(adapter(beliefs, a_base).cpu())

            # Treatment
            a_treat = actions.clone()
            a_treat[:, action_idx] = treatment_val
            all_logit_treat.append(adapter(beliefs, a_treat).cpu())

    logit_base = torch.cat(all_logit_base).numpy()
    logit_treat = torch.cat(all_logit_treat).numpy()
    pred_base = 1.0 / (1.0 + np.exp(-logit_base))
    pred_treat = 1.0 / (1.0 + np.exp(-logit_treat))

    metadata = adapter_dataset.metadata
    result = metadata.copy() if metadata is not None else pd.DataFrame()
    result['pred_baseline'] = pred_base
    result['pred_treatment'] = pred_treat
    result['predicted_uplift'] = pred_treat - pred_base          # 確率スケール
    result['predicted_uplift_logit'] = logit_treat - logit_base  # logitスケール

    return result


# ============================================================
# 8. analyze_heterogeneous_effects
# ============================================================
def analyze_heterogeneous_effects(
    effects_df: pd.DataFrame,
    true_param_col: str = 'true_gamma',
    intervention_col: str = '',
    outcome: str = '',
) -> Dict[str, Any]:
    """
    予測upliftと真のパラメータの関係を分析する。

    確率スケールとlogitスケールの両方でSpearman順位相関を算出。
    logitスケールはsigmoid圧縮を除外するため、潜在パラメータとの
    真の関係をより正確に反映する。
    """
    true_param = effects_df[true_param_col].values

    # --- 確率スケール ---
    uplift_prob = effects_df['predicted_uplift'].values
    rho_prob, p_prob = spearmanr(uplift_prob, true_param)

    q75 = np.percentile(uplift_prob, 75)
    q25 = np.percentile(uplift_prob, 25)
    high_group = true_param[uplift_prob >= q75]
    low_group = true_param[uplift_prob <= q25]

    # --- logitスケール ---
    uplift_logit = effects_df['predicted_uplift_logit'].values
    rho_logit, p_logit = spearmanr(uplift_logit, true_param)

    q75_l = np.percentile(uplift_logit, 75)
    q25_l = np.percentile(uplift_logit, 25)
    high_group_l = true_param[uplift_logit >= q75_l]
    low_group_l = true_param[uplift_logit <= q25_l]

    results = {
        # 確率スケール
        'spearman_rho_prob': rho_prob,
        'spearman_p_value_prob': p_prob,
        'high_uplift_prob_mean': float(np.mean(high_group)) if len(high_group) > 0 else float('nan'),
        'low_uplift_prob_mean': float(np.mean(low_group)) if len(low_group) > 0 else float('nan'),
        # logitスケール
        'spearman_rho_logit': rho_logit,
        'spearman_p_value_logit': p_logit,
        'high_uplift_logit_mean': float(np.mean(high_group_l)) if len(high_group_l) > 0 else float('nan'),
        'low_uplift_logit_mean': float(np.mean(low_group_l)) if len(low_group_l) > 0 else float('nan'),
        'high_uplift_n': len(high_group_l),
        'low_uplift_n': len(low_group_l),
    }

    header = f"Heterogeneous Effects: {intervention_col} -> {outcome}" if intervention_col and outcome else "Heterogeneous Effects Analysis"
    print(f"\n--- {header} ({true_param_col}) ---")
    print(f"  [Probability scale]")
    print(f"    Spearman rho: {rho_prob:.4f} (p={p_prob:.4g})")
    print(f"    High uplift (top 25%): mean {true_param_col} = {results['high_uplift_prob_mean']:.4f}")
    print(f"    Low uplift (bottom 25%): mean {true_param_col} = {results['low_uplift_prob_mean']:.4f}")
    print(f"  [Logit scale]")
    print(f"    Spearman rho: {rho_logit:.4f} (p={p_logit:.4g})")
    print(f"    High uplift (top 25%): mean {true_param_col} = {results['high_uplift_logit_mean']:.4f}")
    print(f"    Low uplift (bottom 25%): mean {true_param_col} = {results['low_uplift_logit_mean']:.4f}")

    return results


# ============================================================
# 9. run_adapter_pipeline
# ============================================================
def run_adapter_pipeline(
    dbm: DeepBoltzmannMachine,
    train_dataset,  # SimulationDataset
    val_dataset,    # SimulationDataset (early stopping & threshold選択)
    test_dataset,   # SimulationDataset
    device: torch.device,
    hidden_dim: int = 64,
    n_epochs: int = 50,
    lr: float = 1e-3,
    dropout: float = 0.1,
    batch_size: int = 256,
    save_dir: Path = None,
    patience: int = 10,
    n_iter: int = 10,
    use_lrdbm: bool = False,
    use_wandb: bool = False,
    intervention_column: str = "promo",
    forward: int = 0,
    use_pos_weight: bool = False,
    pretrained_adapters: Dict[str, 'PredictionAdapter'] = None,
    verbose: int = 2,
) -> Dict[str, Any]:
    """
    全体オーケストレーション:

    1. DBM凍結
    2. belief抽出 (top_layer=True / False) x (train / val / test)
    3. Visit Adapter x 2種を学習・評価
    4. Purchase Adapter x 2種を学習・評価
    5. 介入効果分析
    6. 異質性効果分析
    7. 結果をまとめて返す + wandb記録

    pretrained_adaptersが渡された場合、学習をスキップして評価のみ行う。
    キー形式: "adapter_{task}_{belief_label}" (e.g. "adapter_visit_top")
    """
    skip_training = pretrained_adapters is not None and len(pretrained_adapters) > 0

    if verbose >= 0:
        print("\n" + "=" * 70)
        if skip_training:
            print("PHASE 3: ADAPTER EVALUATION (loaded, training skipped)")
        else:
            print("PHASE 3: ADAPTER TRAINING (Prediction Heads)")
        if forward > 0:
            print(f"  forward={forward}: predicting t+{forward} outcomes from t beliefs")
        print("=" * 70)

    results = {}

    # 1. DBM凍結
    dbm.eval()
    for p in dbm.parameters():
        p.requires_grad = False
    dbm.to(device)

    # 2. belief抽出
    extract_fn = extract_beliefs_lrdbm if use_lrdbm else extract_beliefs

    belief_variants = {}
    for use_top in [True, False]:
        label = "top" if use_top else "full"
        if verbose >= 1:
            print(f"\n  Extracting beliefs (use_top_layer={use_top})...")

        train_adapter_ds = extract_fn(dbm, train_dataset, device,
                                       use_top_layer=use_top, n_iter=n_iter,
                                       forward=forward)
        val_adapter_ds = extract_fn(dbm, val_dataset, device,
                                     use_top_layer=use_top, n_iter=n_iter,
                                     forward=forward)
        test_adapter_ds = extract_fn(dbm, test_dataset, device,
                                      use_top_layer=use_top, n_iter=n_iter,
                                      forward=forward)
        belief_dim = train_adapter_ds.beliefs.shape[1]
        action_dim = train_adapter_ds.actions.shape[1]
        if verbose >= 1:
            print(f"    belief_dim={belief_dim}, action_dim={action_dim}, "
                  f"train={len(train_adapter_ds)}, val={len(val_adapter_ds)}, test={len(test_adapter_ds)}")
        belief_variants[label] = (train_adapter_ds, val_adapter_ds, test_adapter_ds, belief_dim, action_dim)

    # 3 & 4. Adapter学習（or ロード済み復元）・評価
    best_adapters = {}  # task -> (adapter, belief_label, val_metrics)

    for belief_label, (train_ads, val_ads, test_ads, belief_dim, action_dim) in belief_variants.items():
        for task in ['visit', 'purchase']:
            adapter_name = f"{task}_{belief_label}"

            if skip_training:
                # ロード済みadapterを復元
                adapter_key = f"adapter_{adapter_name}"
                if adapter_key not in pretrained_adapters:
                    if verbose >= 1:
                        print(f"\n  Skipping {adapter_name} (not found in loaded adapters)")
                    continue
                adapter = pretrained_adapters[adapter_key].to(device)
                if verbose >= 0:
                    print(f"\n## Evaluating Adapter: {adapter_name} (loaded)")
            else:
                if verbose >= 0:
                    print(f"\n## Training Adapter: {adapter_name}")

                adapter = PredictionAdapter(
                    belief_dim=belief_dim,
                    action_dim=action_dim,
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                    task_name=adapter_name,
                )

                prefix = f"adapter/{adapter_name}/" if use_wandb else ""
                adapter, train_hist = train_adapter(
                    adapter, train_ads, val_ads, task,
                    n_epochs=n_epochs, lr=lr, batch_size=batch_size,
                    device=device, patience=patience,
                    use_wandb=use_wandb, wandb_prefix=prefix,
                    use_pos_weight=use_pos_weight,
                    verbose=verbose,
                )

            # Valでthreshold決定 → train/testに適用
            val_metrics = evaluate_adapter(adapter, val_ads, task, device)
            thresh = val_metrics['best_threshold']
            train_metrics = evaluate_adapter(adapter, train_ads, task, device, threshold=thresh)
            test_metrics = evaluate_adapter(adapter, test_ads, task, device, threshold=thresh)
            if verbose >= 0:
                for split_name, m in [("Train", train_metrics), ("Val", val_metrics), ("Test", test_metrics)]:
                    print(f"  [{adapter_name}] {split_name:5s}: "
                          f"AUC={m['auc']:.4f}, "
                          f"Acc={m['accuracy']:.4f}, "
                          f"F1={m['f1']:.4f}, "
                          f"Prec={m['precision']:.4f}, "
                          f"Rec={m['recall']:.4f}, "
                          f"Thresh={m['best_threshold']:.2f}")

            results[adapter_name] = {
                'train': train_metrics,
                'val': val_metrics,
                'test': test_metrics,
            }

            # wandb summary
            if use_wandb:
                import wandb
                for split_name, m in [("train", train_metrics), ("val", val_metrics), ("test", test_metrics)]:
                    for k, v in m.items():
                        wandb.run.summary[f"adapter/{adapter_name}/{split_name}_{k}"] = v

            # ベストモデルの選択（val AUC基準）
            val_auc = val_metrics['auc']
            if task not in best_adapters or val_auc > best_adapters[task][2]['auc']:
                best_adapters[task] = (adapter, belief_label, val_metrics)

    # 5 & 6. 介入効果分析（ベストadapterを使用）
    # Z_tはadapterの入力なので、action_columnsを使う
    action_cols = test_dataset.action_columns

    # intervention_column → visit
    if 'visit' in best_adapters and intervention_column in action_cols:
        visit_adapter, visit_bl, _ = best_adapters['visit']
        _, _, test_ads, _, _ = belief_variants[visit_bl]
        print(f"\n  Intervention analysis: {intervention_column} -> visit (belief={visit_bl})")

        effects_visit = predict_intervention_effects(
            dbm, visit_adapter, test_ads, action_cols,
            intervention_col=intervention_column,
            baseline_val=0.0, treatment_val=1.0,
            device=device,
        )
        for param_col in ['true_gamma', 'true_alpha', 'true_beta']:
            het = analyze_heterogeneous_effects(effects_visit, param_col,
                                                intervention_col=intervention_column, outcome='visit')
            results[f'intervention_visit_{param_col}'] = het
            if use_wandb:
                import wandb
                for k, v in het.items():
                    wandb.run.summary[f"adapter/intervention_visit/{param_col}/{k}"] = v

    # sale1 → purchase
    if 'purchase' in best_adapters and 'sale1' in action_cols:
        purch_adapter, purch_bl, _ = best_adapters['purchase']
        _, _, test_ads, _, _ = belief_variants[purch_bl]
        print(f"\n  Intervention analysis: sale1 -> purchase (belief={purch_bl})")

        effects_purchase = predict_intervention_effects(
            dbm, purch_adapter, test_ads, action_cols,
            intervention_col='sale1',
            baseline_val=0.0, treatment_val=1.0,
            device=device,
        )
        for param_col in ['true_alpha', 'true_gamma', 'true_beta']:
            het = analyze_heterogeneous_effects(effects_purchase, param_col,
                                                intervention_col='sale1', outcome='purchase')
            results[f'intervention_purchase_{param_col}'] = het
            if use_wandb:
                import wandb
                for k, v in het.items():
                    wandb.run.summary[f"adapter/intervention_purchase/{param_col}/{k}"] = v

    # Adapter保存（ロード済みの場合は再保存しない）
    if save_dir is not None and not skip_training:
        models_dir = Path(save_dir) / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        for task, (adapter, belief_label, _) in best_adapters.items():
            adapter_path = models_dir / f"adapter_{task}_{belief_label}.pkl"
            with open(adapter_path, "wb") as fp:
                pickle.dump(adapter.cpu(), fp)
            print(f"  Saved {adapter_path}")

    print("\n" + "=" * 70)
    print("PHASE 3 COMPLETED")
    print("=" * 70)

    return results


# ============================================================
# 10. Baseline MLP (raw features, no DBM)
# ============================================================
def extract_raw_features(
    dataset,  # SimulationDataset
    forward: int = 0,
) -> AdapterDataset:
    """
    DBMを経由せず、生の特徴量をbelief代わりにAdapterDatasetとして返す。
    Baseline比較用。

    Args:
        dataset: SimulationDataset
        forward: t期の特徴量と t+forward 期のターゲットをペアにする（0なら同期）
    """
    features = dataset.data  # (N, n_visible) — DBM visible units と同じ
    actions = dataset.get_actions()
    targets = dataset.get_targets()
    metadata = dataset.get_metadata()

    if forward == 0:
        return AdapterDataset(features, actions,
                              targets['visit'], targets['purchase'], metadata)

    # forward > 0: t期の特徴量と t+forward 期のターゲットをペアにする
    df = dataset.df.reset_index(drop=True)
    lookup = {}
    for idx, row in df.iterrows():
        lookup[(int(row['consumer_id']), int(row['day']))] = idx

    keep_feat_indices = []
    keep_target_indices = []
    for i in range(len(df)):
        cid = int(df.iloc[i]['consumer_id'])
        t = int(df.iloc[i]['day'])
        target_key = (cid, t + forward)
        if target_key in lookup:
            keep_feat_indices.append(i)
            keep_target_indices.append(lookup[target_key])

    return AdapterDataset(
        features[keep_feat_indices],
        actions[keep_feat_indices],
        targets['visit'][keep_target_indices],
        targets['purchase'][keep_target_indices],
        metadata.iloc[keep_target_indices].reset_index(drop=True),
    )


def run_baseline_mlp(
    train_dataset,  # SimulationDataset
    val_dataset,    # SimulationDataset (early stopping & threshold選択)
    test_dataset,   # SimulationDataset
    device: torch.device,
    hidden_dim: int = 64,
    n_epochs: int = 50,
    lr: float = 1e-3,
    dropout: float = 0.1,
    batch_size: int = 256,
    patience: int = 10,
    forward: int = 0,
    use_pos_weight: bool = False,
    save_dir: Path = None,
    pretrained_baselines: Dict[str, 'PredictionAdapter'] = None,
    intervention_column: str = "push",
    verbose: int = 2,
) -> Dict[str, Any]:
    """
    Baseline MLP: 生の特徴量 + actions → visit/purchase を直接予測。

    DBMのbelief抽出を完全にバイパスし、同じMLP構造・同じ学習設定で
    比較することで、DBM表現の情報損失を診断する。

    - Early stoppingはvalデータで実施
    - ThresholdはvalデータでF1最大化により決定し、train/testに適用

    pretrained_baselinesが渡された場合、学習をスキップして評価のみ行う。
    キー形式: "baseline_{task}" (e.g. "baseline_visit")

    verbose: 0=early stopping以降のみ, 1=epoch log, 2=全て
    """
    skip_training = pretrained_baselines is not None and len(pretrained_baselines) > 0

    if verbose >= 0:
        print("\n" + "=" * 70)
        if skip_training:
            print("BASELINE: MLP EVALUATION (loaded, training skipped)")
        else:
            print("BASELINE: MLP on Raw Features (no DBM)")
        if forward > 0:
            print(f"  forward={forward}: predicting t+{forward} outcomes from t features")
        print("=" * 70)

    # 生の特徴量を抽出
    train_ads = extract_raw_features(train_dataset, forward=forward)
    val_ads = extract_raw_features(val_dataset, forward=forward)
    test_ads = extract_raw_features(test_dataset, forward=forward)

    feat_dim = train_ads.beliefs.shape[1]
    action_dim = train_ads.actions.shape[1]
    if verbose >= 1:
        print(f"  feature_dim={feat_dim}, action_dim={action_dim}, "
              f"train={len(train_ads)}, val={len(val_ads)}, test={len(test_ads)}")

    results = {}
    trained_models = {}  # adapter_name -> PredictionAdapter (保存用)

    for task in ['visit', 'purchase']:
        adapter_name = f"baseline_{task}"

        if skip_training:
            if adapter_name not in pretrained_baselines:
                if verbose >= 1:
                    print(f"\n  Skipping {adapter_name} (not found in loaded baselines)")
                continue
            adapter = pretrained_baselines[adapter_name].to(device)
            if verbose >= 0:
                print(f"\n## Evaluating Baseline: {adapter_name} (loaded)")
        else:
            if verbose >= 0:
                print(f"\n## Training Baseline: {adapter_name}")

            adapter = PredictionAdapter(
                belief_dim=feat_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
                task_name=adapter_name,
            )

            adapter, train_hist = train_adapter(
                adapter, train_ads, val_ads, task,
                n_epochs=n_epochs, lr=lr, batch_size=batch_size,
                device=device, patience=patience,
                use_pos_weight=use_pos_weight,
                verbose=verbose,
            )
            trained_models[adapter_name] = adapter

        # Valでthreshold決定 → train/testに適用
        val_metrics = evaluate_adapter(adapter, val_ads, task, device)
        thresh = val_metrics['best_threshold']
        train_metrics = evaluate_adapter(adapter, train_ads, task, device, threshold=thresh)
        test_metrics = evaluate_adapter(adapter, test_ads, task, device, threshold=thresh)
        if verbose >= 0:
            for split_name, m in [("Train", train_metrics), ("Val", val_metrics), ("Test", test_metrics)]:
                print(f"  [{adapter_name}] {split_name:5s}: "
                      f"AUC={m['auc']:.4f}, "
                      f"Acc={m['accuracy']:.4f}, "
                      f"F1={m['f1']:.4f}, "
                      f"Prec={m['precision']:.4f}, "
                      f"Rec={m['recall']:.4f}, "
                      f"Thresh={m['best_threshold']:.2f}")

        results[adapter_name] = {
            'train': train_metrics,
            'val': val_metrics,
            'test': test_metrics,
        }

    # S-learner CATE分析（Baseline MLPによる反事実予測）
    action_cols = test_dataset.action_columns
    all_baselines = {**trained_models}
    if skip_training and pretrained_baselines:
        all_baselines.update(pretrained_baselines)

    if verbose >= 0:
        print("\n" + "-" * 50)
        print("S-LEARNER CATE ANALYSIS (Baseline MLP)")
        print("-" * 50)

    # intervention_column → visit
    if 'baseline_visit' in all_baselines and intervention_column in action_cols:
        visit_model = all_baselines['baseline_visit'].to(device)
        print(f"\n  S-learner intervention: {intervention_column} -> visit")

        effects_visit = predict_intervention_effects(
            None, visit_model, test_ads, action_cols,
            intervention_col=intervention_column,
            baseline_val=0.0, treatment_val=1.0,
            device=device,
        )
        for param_col in ['true_gamma', 'true_alpha', 'true_beta']:
            het = analyze_heterogeneous_effects(effects_visit, param_col,
                                                intervention_col=intervention_column, outcome='visit')
            results[f'slearner_visit_{param_col}'] = het

    # sale1 → purchase
    if 'baseline_purchase' in all_baselines and 'sale1' in action_cols:
        purch_model = all_baselines['baseline_purchase'].to(device)
        print(f"\n  S-learner intervention: sale1 -> purchase")

        effects_purchase = predict_intervention_effects(
            None, purch_model, test_ads, action_cols,
            intervention_col='sale1',
            baseline_val=0.0, treatment_val=1.0,
            device=device,
        )
        for param_col in ['true_alpha', 'true_gamma', 'true_beta']:
            het = analyze_heterogeneous_effects(effects_purchase, param_col,
                                                intervention_col='sale1', outcome='purchase')
            results[f'slearner_purchase_{param_col}'] = het

    # Baseline保存（ロード済みの場合は再保存しない）
    if save_dir is not None and not skip_training:
        models_dir = Path(save_dir) / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        for name, model in trained_models.items():
            path = models_dir / f"{name}.pkl"
            with open(path, "wb") as fp:
                pickle.dump(model.cpu(), fp)
            if verbose >= 1:
                print(f"  Saved {path}")

    return results


# ============================================================
# 11. T-learner MLP
# ============================================================

def _split_dataset_by_treatment(
    dataset: AdapterDataset,
    action_columns: List[str],
    intervention_col: str,
) -> Tuple[AdapterDataset, AdapterDataset, List[int]]:
    """
    AdapterDatasetを施策変数の値で treated/control に分割し、
    intervention列をactionsから除外する。

    Returns:
        treated_ds: intervention_col == 1 のサブセット（actions から intervention列除外済み）
        control_ds: intervention_col == 0 のサブセット（同上）
        keep_cols: 残したactionsカラムのインデックスリスト
    """
    action_idx = action_columns.index(intervention_col)
    treatment = dataset.actions[:, action_idx]

    treated_mask = treatment == 1.0
    control_mask = treatment == 0.0

    keep_cols = [i for i in range(dataset.actions.shape[1]) if i != action_idx]

    def _subset(mask):
        meta = None
        if dataset.metadata is not None:
            meta = dataset.metadata.iloc[mask.numpy()].reset_index(drop=True)
        return AdapterDataset(
            beliefs=dataset.beliefs[mask],
            actions=dataset.actions[mask][:, keep_cols] if len(keep_cols) > 0
                    else torch.zeros(mask.sum().item(), 0),
            visit=dataset.visit[mask],
            purchase=dataset.purchase[mask],
            metadata=meta,
        )

    return _subset(treated_mask), _subset(control_mask), keep_cols


def _predict_tlearner_cate(
    model_treat: PredictionAdapter,
    model_control: PredictionAdapter,
    test_dataset: AdapterDataset,
    action_columns: List[str],
    intervention_col: str,
    device: torch.device,
    batch_size: int = 256,
) -> pd.DataFrame:
    """
    T-learnerによるCATE予測。

    全テストデータに対して treatment/control 両モデルで予測し、
    差分をupliftとして返す。actions から intervention列を除外して予測する。
    """
    action_idx = action_columns.index(intervention_col)
    keep_cols = [i for i in range(test_dataset.actions.shape[1]) if i != action_idx]

    actions_no_interv = test_dataset.actions[:, keep_cols] if len(keep_cols) > 0 \
        else torch.zeros(len(test_dataset), 0)

    temp_ds = AdapterDataset(
        beliefs=test_dataset.beliefs,
        actions=actions_no_interv,
        visit=test_dataset.visit,
        purchase=test_dataset.purchase,
        metadata=test_dataset.metadata,
    )

    loader = DataLoader(temp_ds, batch_size=batch_size, shuffle=False)

    all_logit_treat = []
    all_logit_control = []

    model_treat.eval()
    model_control.eval()

    with torch.no_grad():
        for batch in loader:
            beliefs = batch['belief'].to(device)
            actions = batch['actions'].to(device)
            all_logit_treat.append(model_treat(beliefs, actions).cpu())
            all_logit_control.append(model_control(beliefs, actions).cpu())

    logit_treat = torch.cat(all_logit_treat).numpy()
    logit_control = torch.cat(all_logit_control).numpy()
    pred_treat = 1.0 / (1.0 + np.exp(-logit_treat))
    pred_control = 1.0 / (1.0 + np.exp(-logit_control))

    metadata = test_dataset.metadata
    result = metadata.copy() if metadata is not None else pd.DataFrame()
    result['pred_baseline'] = pred_control
    result['pred_treatment'] = pred_treat
    result['predicted_uplift'] = pred_treat - pred_control
    result['predicted_uplift_logit'] = logit_treat - logit_control

    return result


def run_tlearner_mlp(
    train_dataset,    # SimulationDataset
    val_dataset,      # SimulationDataset
    test_dataset,     # SimulationDataset
    device: torch.device,
    hidden_dim: int = 64,
    n_epochs: int = 50,
    lr: float = 1e-3,
    dropout: float = 0.1,
    batch_size: int = 256,
    patience: int = 10,
    forward: int = 0,
    use_pos_weight: bool = False,
    save_dir: Path = None,
    intervention_column: str = "push",
    verbose: int = 2,
) -> Dict[str, Any]:
    """
    T-learner MLP: 施策変数で treated/control に分割し、
    それぞれ独立のMLPを学習してCATEを推定する。

    S-learnerとの違い:
    - S-learner: 単一モデル μ(x, z) で CATE = μ(x,1) - μ(x,0)
    - T-learner: 2モデル μ_1(x), μ_0(x) で CATE = μ_1(x) - μ_0(x)
      （施策変数は入力に含めず、データ分割で条件付け）

    各 (intervention, outcome) ペアについて:
    1. 学習/検証データを treated/control に分割
    2. 各群で独立にMLP学習（Early stopping付き）
    3. テストデータ全体で両モデル予測 → CATE算出
    4. Spearman相関による異質性効果分析
    """
    if verbose >= 0:
        print("\n" + "=" * 70)
        print("T-LEARNER: Separate MLPs for Treated/Control")
        if forward > 0:
            print(f"  forward={forward}: predicting t+{forward} outcomes from t features")
        print("=" * 70)

    # 生の特徴量を抽出
    train_ads = extract_raw_features(train_dataset, forward=forward)
    val_ads = extract_raw_features(val_dataset, forward=forward)
    test_ads = extract_raw_features(test_dataset, forward=forward)

    feat_dim = train_ads.beliefs.shape[1]
    action_dim = train_ads.actions.shape[1]
    action_cols = test_dataset.action_columns

    if verbose >= 1:
        print(f"  feature_dim={feat_dim}, action_dim={action_dim}, "
              f"train={len(train_ads)}, val={len(val_ads)}, test={len(test_ads)}")

    results = {}
    trained_models = {}

    # 分析対象の (intervention_col, outcome, true_param) リスト
    analyses = []
    if intervention_column in action_cols:
        analyses.append((intervention_column, 'visit',
                         ['true_gamma', 'true_alpha', 'true_beta']))
    if 'sale1' in action_cols:
        analyses.append(('sale1', 'purchase',
                         ['true_alpha', 'true_gamma', 'true_beta']))

    for interv_col, task, param_cols in analyses:
        if verbose >= 0:
            print(f"\n" + "-" * 50)
            print(f"T-LEARNER: {interv_col} -> {task}")
            print("-" * 50)

        # データ分割（intervention列を除外）
        train_treat, train_ctrl, keep_cols = _split_dataset_by_treatment(
            train_ads, action_cols, interv_col)
        val_treat, val_ctrl, _ = _split_dataset_by_treatment(
            val_ads, action_cols, interv_col)

        reduced_action_dim = len(keep_cols)

        if verbose >= 1:
            print(f"  Train: treated={len(train_treat)}, control={len(train_ctrl)}")
            print(f"  Val:   treated={len(val_treat)}, control={len(val_ctrl)}")
            print(f"  Reduced action_dim={reduced_action_dim} "
                  f"(removed '{interv_col}')")

        # --- Treated model ---
        treat_name = f"tlearner_{task}_{interv_col}_treated"
        if verbose >= 0:
            print(f"\n## Training: {treat_name}")

        model_treat = PredictionAdapter(
            belief_dim=feat_dim,
            action_dim=reduced_action_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            task_name=treat_name,
        )
        model_treat, _ = train_adapter(
            model_treat, train_treat, val_treat, task,
            n_epochs=n_epochs, lr=lr, batch_size=batch_size,
            device=device, patience=patience,
            use_pos_weight=use_pos_weight,
            verbose=verbose,
        )
        trained_models[treat_name] = model_treat

        # --- Control model ---
        ctrl_name = f"tlearner_{task}_{interv_col}_control"
        if verbose >= 0:
            print(f"\n## Training: {ctrl_name}")

        model_ctrl = PredictionAdapter(
            belief_dim=feat_dim,
            action_dim=reduced_action_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            task_name=ctrl_name,
        )
        model_ctrl, _ = train_adapter(
            model_ctrl, train_ctrl, val_ctrl, task,
            n_epochs=n_epochs, lr=lr, batch_size=batch_size,
            device=device, patience=patience,
            use_pos_weight=use_pos_weight,
            verbose=verbose,
        )
        trained_models[ctrl_name] = model_ctrl

        # --- 各群でのPrediction metrics ---
        test_treat, test_ctrl, _ = _split_dataset_by_treatment(
            test_ads, action_cols, interv_col)

        for model, ds, label in [
            (model_treat, test_treat, "treated_on_treated"),
            (model_ctrl, test_ctrl, "control_on_control"),
        ]:
            if len(ds) > 0:
                m = evaluate_adapter(model, ds, task, device)
                if verbose >= 0:
                    print(f"  [{label}] Test: "
                          f"AUC={m['auc']:.4f}, "
                          f"Acc={m['accuracy']:.4f}, "
                          f"F1={m['f1']:.4f}")
                results[f'{treat_name}_{label}'] = m

        # --- CATE prediction ---
        if verbose >= 0:
            print(f"\n  T-learner CATE: {interv_col} -> {task}")

        effects = _predict_tlearner_cate(
            model_treat, model_ctrl, test_ads, action_cols,
            interv_col, device,
        )

        for param_col in param_cols:
            het = analyze_heterogeneous_effects(
                effects, param_col,
                intervention_col=interv_col, outcome=task,
            )
            results[f'tlearner_{task}_{param_col}'] = het

    # モデル保存
    if save_dir is not None:
        models_dir = Path(save_dir) / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        for name, model in trained_models.items():
            path = models_dir / f"{name}.pkl"
            with open(path, "wb") as fp:
                pickle.dump(model.cpu(), fp)
            if verbose >= 1:
                print(f"  Saved {path}")

    if verbose >= 0:
        print("\n" + "=" * 70)
        print("T-LEARNER COMPLETED")
        print("=" * 70)

    return results


# ============================================================
# 12. Causal Forest (CausalForestDML)
# ============================================================

def run_causalforest(
    train_dataset,    # SimulationDataset
    val_dataset,      # SimulationDataset
    test_dataset,     # SimulationDataset
    n_estimators: int = 1000,
    min_samples_leaf: int = 5,
    max_depth: int = None,
    forward: int = 0,
    intervention_column: str = "push",
    verbose: int = 2,
) -> Dict[str, Any]:
    """
    Causal Forest による CATE 推定ベースライン。

    econml.dml.CausalForestDML を使用。
    Train+Val を結合して学習し、Test で CATE を推定する。

    各 (intervention, outcome) ペアについて:
    1. extract_raw_features で生特徴量を取得
    2. Train+Val を結合（CFはearly stoppingなし）
    3. CausalForestDML で学習
    4. Test上で .effect(X_test) → CATE推定値
    5. analyze_heterogeneous_effects で評価
    """
    from econml.dml import CausalForestDML
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    if verbose >= 0:
        print("\n" + "=" * 70)
        print("CAUSAL FOREST: CausalForestDML")
        if forward > 0:
            print(f"  forward={forward}: predicting t+{forward} outcomes from t features")
        print("=" * 70)

    # 生の特徴量を抽出
    train_ads = extract_raw_features(train_dataset, forward=forward)
    val_ads = extract_raw_features(val_dataset, forward=forward)
    test_ads = extract_raw_features(test_dataset, forward=forward)

    # Train+Val を結合
    combined_beliefs = torch.cat([train_ads.beliefs, val_ads.beliefs], dim=0)
    combined_actions = torch.cat([train_ads.actions, val_ads.actions], dim=0)
    combined_visit = torch.cat([train_ads.visit, val_ads.visit], dim=0)
    combined_purchase = torch.cat([train_ads.purchase, val_ads.purchase], dim=0)
    combined_meta = pd.concat(
        [train_ads.metadata, val_ads.metadata], ignore_index=True
    ) if train_ads.metadata is not None else None
    combined_ads = AdapterDataset(
        combined_beliefs, combined_actions,
        combined_visit, combined_purchase, combined_meta,
    )

    action_cols = test_dataset.action_columns

    if verbose >= 1:
        feat_dim = combined_ads.beliefs.shape[1]
        action_dim = combined_ads.actions.shape[1]
        print(f"  feature_dim={feat_dim}, action_dim={action_dim}, "
              f"train+val={len(combined_ads)}, test={len(test_ads)}")

    results = {}

    # 分析対象の (intervention_col, outcome, true_param) リスト
    analyses = []
    if intervention_column in action_cols:
        analyses.append((intervention_column, 'visit',
                         ['true_gamma', 'true_alpha', 'true_beta']))
    if 'sale1' in action_cols:
        analyses.append(('sale1', 'purchase',
                         ['true_alpha', 'true_gamma', 'true_beta']))

    for interv_col, task, param_cols in analyses:
        if verbose >= 0:
            print(f"\n" + "-" * 50)
            print(f"CAUSAL FOREST: {interv_col} -> {task}")
            print("-" * 50)

        # 特徴量を構築: beliefs + actions（介入列を除外）
        action_idx = action_cols.index(interv_col)
        keep_cols = [i for i in range(combined_ads.actions.shape[1]) if i != action_idx]

        X_train = np.hstack([
            combined_ads.beliefs.numpy(),
            combined_ads.actions[:, keep_cols].numpy() if len(keep_cols) > 0
            else np.zeros((len(combined_ads), 0)),
        ])
        T_train = combined_ads.actions[:, action_idx].numpy()
        Y_train = (combined_ads.visit if task == 'visit' else combined_ads.purchase).numpy()

        X_test = np.hstack([
            test_ads.beliefs.numpy(),
            test_ads.actions[:, keep_cols].numpy() if len(keep_cols) > 0
            else np.zeros((len(test_ads), 0)),
        ])

        if verbose >= 1:
            print(f"  X_train: {X_train.shape}, T_train: {T_train.shape}")
            print(f"  X_test: {X_test.shape}")
            print(f"  Treatment rate (train+val): {T_train.mean():.3f}")

        # CausalForestDML の学習
        # T はbinaryなので discrete_treatment=True を指定
        cf = CausalForestDML(
            model_y=RandomForestRegressor(n_estimators=100, min_samples_leaf=5, n_jobs=-1),
            model_t=RandomForestClassifier(n_estimators=100, min_samples_leaf=5, n_jobs=-1),
            discrete_treatment=True,
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            max_depth=max_depth,
            random_state=42,
        )

        if verbose >= 1:
            print(f"  Fitting CausalForestDML (n_estimators={n_estimators})...")

        cf.fit(Y_train, T_train, X=X_train)

        # CATE推定
        cate = cf.effect(X_test)  # (n_test,)

        if verbose >= 1:
            print(f"  CATE stats: mean={cate.mean():.4f}, "
                  f"std={cate.std():.4f}, "
                  f"min={cate.min():.4f}, max={cate.max():.4f}")

        # effects DataFrame を構築
        metadata = test_ads.metadata
        effects = metadata.copy() if metadata is not None else pd.DataFrame()
        effects['predicted_uplift'] = cate
        effects['predicted_uplift_logit'] = cate  # CFはリスク差スケール、logit分解なし

        for param_col in param_cols:
            het = analyze_heterogeneous_effects(
                effects, param_col,
                intervention_col=interv_col, outcome=task,
            )
            results[f'cf_{task}_{param_col}'] = het

    if verbose >= 0:
        print("\n" + "=" * 70)
        print("CAUSAL FOREST COMPLETED")
        print("=" * 70)

    return results


# ============================================================
# 13. X-learner / DR-learner shared helpers
# ============================================================

def _prepare_metalearner_arrays(
    train_dataset,
    val_dataset,
    test_dataset,
    forward: int,
    intervention_column: str,
    verbose: int,
):
    """
    X-learner / DR-learner 共通の前処理。
    Train+Val を結合し、各 (intervention, outcome) ペアに対して
    (X_train, T_train, Y_train, X_test, effects_meta) を返すジェネレータを構築する。
    """
    train_ads = extract_raw_features(train_dataset, forward=forward)
    val_ads = extract_raw_features(val_dataset, forward=forward)
    test_ads = extract_raw_features(test_dataset, forward=forward)

    combined_beliefs = torch.cat([train_ads.beliefs, val_ads.beliefs], dim=0)
    combined_actions = torch.cat([train_ads.actions, val_ads.actions], dim=0)
    combined_visit = torch.cat([train_ads.visit, val_ads.visit], dim=0)
    combined_purchase = torch.cat([train_ads.purchase, val_ads.purchase], dim=0)
    combined_meta = pd.concat(
        [train_ads.metadata, val_ads.metadata], ignore_index=True
    ) if train_ads.metadata is not None else None
    combined_ads = AdapterDataset(
        combined_beliefs, combined_actions,
        combined_visit, combined_purchase, combined_meta,
    )

    action_cols = test_dataset.action_columns

    if verbose >= 1:
        feat_dim = combined_ads.beliefs.shape[1]
        action_dim = combined_ads.actions.shape[1]
        print(f"  feature_dim={feat_dim}, action_dim={action_dim}, "
              f"train+val={len(combined_ads)}, test={len(test_ads)}")

    analyses = []
    if intervention_column in action_cols:
        analyses.append((intervention_column, 'visit',
                         ['true_gamma', 'true_alpha', 'true_beta']))
    if 'sale1' in action_cols:
        analyses.append(('sale1', 'purchase',
                         ['true_alpha', 'true_gamma', 'true_beta']))

    return combined_ads, test_ads, action_cols, analyses


def _build_xy_for_treatment(
    combined_ads: AdapterDataset,
    test_ads: AdapterDataset,
    action_cols: List[str],
    interv_col: str,
    task: str,
):
    """与えられた (intervention, outcome) について X, T, Y, X_test を構築する。"""
    action_idx = action_cols.index(interv_col)
    keep_cols = [i for i in range(combined_ads.actions.shape[1]) if i != action_idx]

    X_train = np.hstack([
        combined_ads.beliefs.numpy(),
        combined_ads.actions[:, keep_cols].numpy() if len(keep_cols) > 0
        else np.zeros((len(combined_ads), 0)),
    ])
    T_train = combined_ads.actions[:, action_idx].numpy().astype(int)
    Y_train = (combined_ads.visit if task == 'visit' else combined_ads.purchase).numpy()

    X_test = np.hstack([
        test_ads.beliefs.numpy(),
        test_ads.actions[:, keep_cols].numpy() if len(keep_cols) > 0
        else np.zeros((len(test_ads), 0)),
    ])
    return X_train, T_train, Y_train, X_test


# ============================================================
# 14. X-learner (econml.metalearners.XLearner)
# ============================================================

def run_xlearner(
    train_dataset,
    val_dataset,
    test_dataset,
    n_estimators: int = 100,
    min_samples_leaf: int = 5,
    forward: int = 0,
    intervention_column: str = "push",
    verbose: int = 2,
) -> Dict[str, Any]:
    """
    X-learner (Künzel et al., 2019) による CATE 推定ベースライン。

    econml.metalearners.XLearner を使用。Train+Val を結合して学習し、
    Test で CATE を推定する。base learner は CausalForest と統一して
    RandomForest を使用する。
    """
    from econml.metalearners import XLearner
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    if verbose >= 0:
        print("\n" + "=" * 70)
        print("X-LEARNER: econml.metalearners.XLearner")
        if forward > 0:
            print(f"  forward={forward}: predicting t+{forward} outcomes from t features")
        print("=" * 70)

    combined_ads, test_ads, action_cols, analyses = _prepare_metalearner_arrays(
        train_dataset, val_dataset, test_dataset,
        forward, intervention_column, verbose,
    )

    results = {}

    for interv_col, task, param_cols in analyses:
        if verbose >= 0:
            print(f"\n" + "-" * 50)
            print(f"X-LEARNER: {interv_col} -> {task}")
            print("-" * 50)

        X_train, T_train, Y_train, X_test = _build_xy_for_treatment(
            combined_ads, test_ads, action_cols, interv_col, task,
        )

        if verbose >= 1:
            print(f"  X_train: {X_train.shape}, T_train: {T_train.shape}")
            print(f"  X_test: {X_test.shape}")
            print(f"  Treatment rate (train+val): {T_train.mean():.3f}")

        xl = XLearner(
            models=RandomForestRegressor(
                n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
                n_jobs=-1, random_state=42,
            ),
            propensity_model=RandomForestClassifier(
                n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
                n_jobs=-1, random_state=42,
            ),
            cate_models=RandomForestRegressor(
                n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
                n_jobs=-1, random_state=42,
            ),
        )

        if verbose >= 1:
            print(f"  Fitting XLearner (n_estimators={n_estimators})...")

        xl.fit(Y_train, T_train, X=X_train)

        cate = xl.effect(X_test)

        if verbose >= 1:
            print(f"  CATE stats: mean={cate.mean():.4f}, "
                  f"std={cate.std():.4f}, "
                  f"min={cate.min():.4f}, max={cate.max():.4f}")

        metadata = test_ads.metadata
        effects = metadata.copy() if metadata is not None else pd.DataFrame()
        effects['predicted_uplift'] = cate
        effects['predicted_uplift_logit'] = cate  # risk-difference scale, no logit decomposition

        for param_col in param_cols:
            het = analyze_heterogeneous_effects(
                effects, param_col,
                intervention_col=interv_col, outcome=task,
            )
            results[f'xl_{task}_{param_col}'] = het

    if verbose >= 0:
        print("\n" + "=" * 70)
        print("X-LEARNER COMPLETED")
        print("=" * 70)

    return results


# ============================================================
# 15. DR-learner (econml.dr.DRLearner)
# ============================================================

def run_drlearner(
    train_dataset,
    val_dataset,
    test_dataset,
    n_estimators: int = 100,
    min_samples_leaf: int = 5,
    cv: int = 2,
    forward: int = 0,
    intervention_column: str = "push",
    verbose: int = 2,
) -> Dict[str, Any]:
    """
    DR-learner (Kennedy, 2020) による CATE 推定ベースライン。

    econml.dr.DRLearner を使用。outcome / propensity を cross-fit で推定し、
    doubly-robust pseudo-outcome を最終モデルで回帰する。base learner は
    CausalForest / X-learner と統一して RandomForest を使用する。
    """
    from econml.dr import DRLearner
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    if verbose >= 0:
        print("\n" + "=" * 70)
        print("DR-LEARNER: econml.dr.DRLearner")
        if forward > 0:
            print(f"  forward={forward}: predicting t+{forward} outcomes from t features")
        print("=" * 70)

    combined_ads, test_ads, action_cols, analyses = _prepare_metalearner_arrays(
        train_dataset, val_dataset, test_dataset,
        forward, intervention_column, verbose,
    )

    results = {}

    for interv_col, task, param_cols in analyses:
        if verbose >= 0:
            print(f"\n" + "-" * 50)
            print(f"DR-LEARNER: {interv_col} -> {task}")
            print("-" * 50)

        X_train, T_train, Y_train, X_test = _build_xy_for_treatment(
            combined_ads, test_ads, action_cols, interv_col, task,
        )

        if verbose >= 1:
            print(f"  X_train: {X_train.shape}, T_train: {T_train.shape}")
            print(f"  X_test: {X_test.shape}")
            print(f"  Treatment rate (train+val): {T_train.mean():.3f}")

        dr = DRLearner(
            model_propensity=RandomForestClassifier(
                n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
                n_jobs=-1, random_state=42,
            ),
            model_regression=RandomForestRegressor(
                n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
                n_jobs=-1, random_state=42,
            ),
            model_final=RandomForestRegressor(
                n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
                n_jobs=-1, random_state=42,
            ),
            cv=cv,
            random_state=42,
        )

        if verbose >= 1:
            print(f"  Fitting DRLearner (n_estimators={n_estimators}, cv={cv})...")

        dr.fit(Y_train, T_train, X=X_train)

        cate = dr.effect(X_test)

        if verbose >= 1:
            print(f"  CATE stats: mean={cate.mean():.4f}, "
                  f"std={cate.std():.4f}, "
                  f"min={cate.min():.4f}, max={cate.max():.4f}")

        metadata = test_ads.metadata
        effects = metadata.copy() if metadata is not None else pd.DataFrame()
        effects['predicted_uplift'] = cate
        effects['predicted_uplift_logit'] = cate  # risk-difference scale, no logit decomposition

        for param_col in param_cols:
            het = analyze_heterogeneous_effects(
                effects, param_col,
                intervention_col=interv_col, outcome=task,
            )
            results[f'dr_{task}_{param_col}'] = het

    if verbose >= 0:
        print("\n" + "=" * 70)
        print("DR-LEARNER COMPLETED")
        print("=" * 70)

    return results


# ============================================================
# 16. TARNet (Shalit et al., 2017)
# ============================================================

class _TARNet(nn.Module):
    """共有 representation Φ(X) と treatment-specific outcome head h_0, h_1。"""

    def __init__(self, input_dim: int, repr_dim: int = 64,
                 head_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        self.repr = nn.Sequential(
            nn.Linear(input_dim, repr_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(repr_dim, repr_dim), nn.ReLU(), nn.Dropout(dropout),
        )
        self.h0 = nn.Sequential(
            nn.Linear(repr_dim, head_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(head_dim, 1),
        )
        self.h1 = nn.Sequential(
            nn.Linear(repr_dim, head_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(head_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        phi = self.repr(x)
        return self.h0(phi).squeeze(-1), self.h1(phi).squeeze(-1)


def _train_tarnet(
    model: _TARNet,
    X_train: np.ndarray, T_train: np.ndarray, Y_train: np.ndarray,
    X_val: np.ndarray, T_val: np.ndarray, Y_val: np.ndarray,
    device: torch.device,
    n_epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 256,
    patience: int = 10,
    verbose: int = 2,
    tag: str = "tarnet",
) -> _TARNet:
    """TARNet を factual loss (treatment-stratified BCE) で学習。Val factual loss で early stopping。"""
    model.to(device)
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    X_train_t = torch.from_numpy(X_train).float()
    T_train_t = torch.from_numpy(T_train).float()
    Y_train_t = torch.from_numpy(Y_train).float()
    X_val_t = torch.from_numpy(X_val).float().to(device)
    T_val_t = torch.from_numpy(T_val).float().to(device)
    Y_val_t = torch.from_numpy(Y_val).float().to(device)

    n = len(X_train_t)
    best_val = float('inf')
    best_state = None
    epochs_no_improve = 0

    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(n)
        total_loss, n_seen = 0.0, 0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            xb = X_train_t[idx].to(device)
            tb = T_train_t[idx].to(device)
            yb = Y_train_t[idx].to(device)

            logit0, logit1 = model(xb)
            loss0 = criterion(logit0, yb) * (1.0 - tb)
            loss1 = criterion(logit1, yb) * tb
            loss = (loss0 + loss1).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(idx)
            n_seen += len(idx)
        train_loss = total_loss / max(n_seen, 1)

        # --- Val factual loss ---
        model.eval()
        with torch.no_grad():
            logit0, logit1 = model(X_val_t)
            l0 = criterion(logit0, Y_val_t) * (1.0 - T_val_t)
            l1 = criterion(logit1, Y_val_t) * T_val_t
            val_loss = (l0 + l1).mean().item()

        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
            star = ' *'
        else:
            epochs_no_improve += 1
            star = ''

        if verbose >= 1 and (epoch % 10 == 0 or star):
            print(f"  [{tag}] Epoch {epoch}: train_loss={train_loss:.4f}, "
                  f"val_loss={val_loss:.4f}{star}")

        if epochs_no_improve >= patience:
            if verbose >= 0:
                print(f"  [{tag}] Early stopping at epoch {epoch} (best val_loss={best_val:.4f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)
    return model


def run_tarnet(
    train_dataset,
    val_dataset,
    test_dataset,
    device: torch.device,
    repr_dim: int = 64,
    head_dim: int = 32,
    n_epochs: int = 50,
    lr: float = 1e-3,
    dropout: float = 0.1,
    batch_size: int = 256,
    patience: int = 10,
    forward: int = 0,
    intervention_column: str = "push",
    verbose: int = 2,
) -> Dict[str, Any]:
    """
    TARNet (Shalit et al., 2017) による CATE 推定ベースライン。

    共有 representation Φ(X) の上に treatment-specific な 2 個の outcome head を持ち、
    factual loss (各 unit はその unit が実際に受けた treatment の head のみで学習) で
    最適化する。CATE = σ(h_1(Φ(x))) − σ(h_0(Φ(x)))（probability スケール）と
    h_1(Φ(x)) − h_0(Φ(x))（logit スケール）の両方を返す。
    """
    if verbose >= 0:
        print("\n" + "=" * 70)
        print("TARNET: shared trunk + 2 outcome heads")
        if forward > 0:
            print(f"  forward={forward}: predicting t+{forward} outcomes from t features")
        print("=" * 70)

    train_ads = extract_raw_features(train_dataset, forward=forward)
    val_ads = extract_raw_features(val_dataset, forward=forward)
    test_ads = extract_raw_features(test_dataset, forward=forward)

    action_cols = test_dataset.action_columns

    if verbose >= 1:
        feat_dim = train_ads.beliefs.shape[1]
        action_dim = train_ads.actions.shape[1]
        print(f"  feature_dim={feat_dim}, action_dim={action_dim}, "
              f"train={len(train_ads)}, val={len(val_ads)}, test={len(test_ads)}")

    analyses = []
    if intervention_column in action_cols:
        analyses.append((intervention_column, 'visit',
                         ['true_gamma', 'true_alpha', 'true_beta']))
    if 'sale1' in action_cols:
        analyses.append(('sale1', 'purchase',
                         ['true_alpha', 'true_gamma', 'true_beta']))

    results = {}

    for interv_col, task, param_cols in analyses:
        if verbose >= 0:
            print(f"\n" + "-" * 50)
            print(f"TARNET: {interv_col} -> {task}")
            print("-" * 50)

        action_idx = action_cols.index(interv_col)
        keep_cols = [i for i in range(train_ads.actions.shape[1]) if i != action_idx]

        def _build(ads):
            X = np.hstack([
                ads.beliefs.numpy(),
                ads.actions[:, keep_cols].numpy() if len(keep_cols) > 0
                else np.zeros((len(ads), 0)),
            ])
            T = ads.actions[:, action_idx].numpy()
            Y = (ads.visit if task == 'visit' else ads.purchase).numpy()
            return X.astype(np.float32), T.astype(np.float32), Y.astype(np.float32)

        X_train, T_train, Y_train = _build(train_ads)
        X_val, T_val, Y_val = _build(val_ads)
        X_test, _, _ = _build(test_ads)

        if verbose >= 1:
            print(f"  X_train: {X_train.shape}, treatment rate: {T_train.mean():.3f}")

        model = _TARNet(input_dim=X_train.shape[1], repr_dim=repr_dim,
                        head_dim=head_dim, dropout=dropout)
        model = _train_tarnet(
            model, X_train, T_train, Y_train, X_val, T_val, Y_val, device,
            n_epochs=n_epochs, lr=lr, batch_size=batch_size, patience=patience,
            verbose=verbose, tag=f"tarnet_{task}_{interv_col}",
        )

        # CATE on test
        model.eval()
        with torch.no_grad():
            X_test_t = torch.from_numpy(X_test).to(device)
            logit0, logit1 = model(X_test_t)
            logit0 = logit0.cpu().numpy()
            logit1 = logit1.cpu().numpy()
        prob0 = 1.0 / (1.0 + np.exp(-logit0))
        prob1 = 1.0 / (1.0 + np.exp(-logit1))

        cate_logit = logit1 - logit0
        cate_prob = prob1 - prob0

        if verbose >= 1:
            print(f"  CATE (prob) stats: mean={cate_prob.mean():.4f}, "
                  f"std={cate_prob.std():.4f}")
            print(f"  CATE (logit) stats: mean={cate_logit.mean():.4f}, "
                  f"std={cate_logit.std():.4f}")

        metadata = test_ads.metadata
        effects = metadata.copy() if metadata is not None else pd.DataFrame()
        effects['predicted_uplift'] = cate_prob
        effects['predicted_uplift_logit'] = cate_logit

        for param_col in param_cols:
            het = analyze_heterogeneous_effects(
                effects, param_col,
                intervention_col=interv_col, outcome=task,
            )
            results[f'tarnet_{task}_{param_col}'] = het

    if verbose >= 0:
        print("\n" + "=" * 70)
        print("TARNET COMPLETED")
        print("=" * 70)

    return results


# ============================================================
# 17. CEVAE (Louizos et al., 2017) — pyro.contrib.cevae wrapper
# ============================================================

def run_cevae(
    train_dataset,
    val_dataset,
    test_dataset,
    latent_dim: int = 16,
    hidden_dim: int = 128,
    num_layers: int = 3,
    num_samples: int = 100,
    n_epochs: int = 30,
    lr: float = 1e-3,
    batch_size: int = 256,
    forward: int = 0,
    intervention_column: str = "push",
    verbose: int = 2,
) -> Dict[str, Any]:
    """
    CEVAE (Louizos et al., 2017) による CATE 推定ベースライン。

    pyro.contrib.cevae.CEVAE を使用し、共変量 X からの潜在 Z 経由で
    treatment / outcome の同時分布を学習する。outcome は binary なので
    `outcome_dist='bernoulli'`。`.ite()` から得られる ITE を probability
    スケールの CATE として扱う（logit 列は同値で埋める）。
    """
    from pyro.contrib.cevae import CEVAE

    if verbose >= 0:
        print("\n" + "=" * 70)
        print("CEVAE: pyro.contrib.cevae.CEVAE")
        if forward > 0:
            print(f"  forward={forward}: predicting t+{forward} outcomes from t features")
        print("=" * 70)

    combined_ads, test_ads, action_cols, analyses = _prepare_metalearner_arrays(
        train_dataset, val_dataset, test_dataset,
        forward, intervention_column, verbose,
    )

    results = {}

    for interv_col, task, param_cols in analyses:
        if verbose >= 0:
            print(f"\n" + "-" * 50)
            print(f"CEVAE: {interv_col} -> {task}")
            print("-" * 50)

        X_train, T_train, Y_train, X_test = _build_xy_for_treatment(
            combined_ads, test_ads, action_cols, interv_col, task,
        )
        X_train = X_train.astype(np.float32)
        T_train = T_train.astype(np.float32)
        Y_train = Y_train.astype(np.float32)
        X_test = X_test.astype(np.float32)

        if verbose >= 1:
            print(f"  X_train: {X_train.shape}, T_train: {T_train.shape}")
            print(f"  X_test: {X_test.shape}")
            print(f"  Treatment rate (train+val): {T_train.mean():.3f}")

        cevae = CEVAE(
            feature_dim=X_train.shape[1],
            outcome_dist='bernoulli',
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_samples=num_samples,
        )

        if verbose >= 1:
            print(f"  Fitting CEVAE (latent_dim={latent_dim}, hidden_dim={hidden_dim}, "
                  f"epochs={n_epochs})...")

        cevae.fit(
            torch.from_numpy(X_train),
            torch.from_numpy(T_train),
            torch.from_numpy(Y_train),
            num_epochs=n_epochs,
            batch_size=batch_size,
            learning_rate=lr,
            log_every=max(1, n_epochs // 10),
        )

        ite = cevae.ite(torch.from_numpy(X_test), batch_size=batch_size)
        cate = ite.detach().cpu().numpy()

        if verbose >= 1:
            print(f"  CATE stats: mean={cate.mean():.4f}, std={cate.std():.4f}, "
                  f"min={cate.min():.4f}, max={cate.max():.4f}")

        metadata = test_ads.metadata
        effects = metadata.copy() if metadata is not None else pd.DataFrame()
        effects['predicted_uplift'] = cate
        effects['predicted_uplift_logit'] = cate  # probability-scale only

        for param_col in param_cols:
            het = analyze_heterogeneous_effects(
                effects, param_col,
                intervention_col=interv_col, outcome=task,
            )
            results[f'cevae_{task}_{param_col}'] = het

    if verbose >= 0:
        print("\n" + "=" * 70)
        print("CEVAE COMPLETED")
        print("=" * 70)

    return results


# ============================================================
# Self-test
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("Adapter Self-Test")
    print("=" * 60)

    device = find_device()
    print(f"Using device: {device}")

    # --- Test 1: PredictionAdapter forward/backward ---
    print("\n--- Test 1: PredictionAdapter ---")
    belief_dim = 32
    action_dim = 5
    adapter = PredictionAdapter(belief_dim, action_dim=action_dim,
                                hidden_dim=16, dropout=0.1, task_name="test")
    adapter.to(device)

    dummy_belief = torch.randn(8, belief_dim, device=device)
    dummy_actions = torch.randint(0, 2, (8, action_dim), device=device).float()
    logits = adapter(dummy_belief, dummy_actions)
    assert logits.shape == (8,), f"Expected (8,), got {logits.shape}"

    loss = nn.BCEWithLogitsLoss()(logits, torch.ones(8, device=device))
    loss.backward()
    print(f"  forward shape: {logits.shape}, loss: {loss.item():.4f}")
    print("  Test 1 passed!")

    # --- Test 2: AdapterDataset ---
    print("\n--- Test 2: AdapterDataset ---")
    n = 100
    beliefs = torch.randn(n, belief_dim)
    actions = torch.randint(0, 2, (n, action_dim)).float()
    visit = torch.bernoulli(torch.full((n,), 0.3))
    purchase = torch.bernoulli(torch.full((n,), 0.1))

    ds = AdapterDataset(beliefs, actions, visit, purchase)
    assert len(ds) == n
    sample = ds[0]
    assert 'belief' in sample and 'actions' in sample and 'visit' in sample and 'purchase' in sample
    print(f"  len={len(ds)}, sample keys={list(sample.keys())}")
    print("  Test 2 passed!")

    # --- Test 3: train_adapter ---
    print("\n--- Test 3: train_adapter ---")
    train_ds = AdapterDataset(
        torch.randn(200, belief_dim),
        torch.randint(0, 2, (200, action_dim)).float(),
        torch.bernoulli(torch.full((200,), 0.3)),
        torch.bernoulli(torch.full((200,), 0.1)),
    )
    val_ds = AdapterDataset(
        torch.randn(50, belief_dim),
        torch.randint(0, 2, (50, action_dim)).float(),
        torch.bernoulli(torch.full((50,), 0.3)),
        torch.bernoulli(torch.full((50,), 0.1)),
    )

    adapter_visit = PredictionAdapter(belief_dim, action_dim=action_dim,
                                      hidden_dim=16, task_name="visit_test")
    adapter_visit, hist = train_adapter(
        adapter_visit, train_ds, val_ds, task='visit',
        n_epochs=5, lr=1e-3, batch_size=32, device=device, patience=3,
    )
    assert 'best_val_auc' in hist
    print(f"  best_val_auc: {hist['best_val_auc']:.4f}")
    print("  Test 3 passed!")

    # --- Test 4: evaluate_adapter ---
    print("\n--- Test 4: evaluate_adapter ---")
    metrics = evaluate_adapter(adapter_visit, val_ds, 'visit', device)
    assert 'auc' in metrics and 'accuracy' in metrics and 'f1' in metrics
    print(f"  metrics: { {k: f'{v:.4f}' for k, v in metrics.items()} }")
    print("  Test 4 passed!")

    # --- Test 5: purchase task (visit=1 subset) ---
    print("\n--- Test 5: purchase adapter ---")
    adapter_purch = PredictionAdapter(belief_dim, action_dim=action_dim,
                                      hidden_dim=16, task_name="purchase_test")
    adapter_purch, hist_p = train_adapter(
        adapter_purch, train_ds, val_ds, task='purchase',
        n_epochs=5, lr=1e-3, batch_size=32, device=device, patience=3,
    )
    metrics_p = evaluate_adapter(adapter_purch, val_ds, 'purchase', device)
    print(f"  purchase metrics: { {k: f'{v:.4f}' for k, v in metrics_p.items()} }")
    print("  Test 5 passed!")

    print("\n" + "=" * 60)
    print("All adapter self-tests passed!")
    print("=" * 60)
