"""
Simulation Data Generator for Marketing World Model (v2)

新しい実験設計:
- 日単位の時間解像度
- 施策の多層構造（日付/店舗×日付/個人×日付）
- 施策→価格の因果的合成
- Visit/Purchaseの独立生成

Author: Junichiro Niimi
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Tuple, Dict, Optional, List
import json
from pathlib import Path
import datetime


@dataclass
class SimulationConfig:
    """シミュレーションの設定パラメータ"""
    dataset_id: str = "default_dataset"

    # スケール
    n_consumers: int = 1024
    n_days: int = 365
    n_stores: int = 10
    price_base: float = 100.0
    seed: int = 42
    period_begin: datetime.date = field(
        default_factory=lambda: datetime.date(2025, 1, 1)
    )

    # 店舗割当確率（Noneの場合はデフォルト）
    store_probs: Optional[List[float]] = None

    # 消費者の異質性
    alpha_mean: float = 2.0        # 価格感度の平均
    # alpha_std: float = 0.5         # 価格感度の標準偏差
    alpha_std: float = 0.3           # 価格感度の標準偏差
    # alpha_income_effect: float = -0.35  # 所得による価格感度の調整（高所得→低感度）
    alpha_income_effect: float = -0.6    # 所得による価格感度の調整（高所得→低感度）
    # alpha_age_effect: float = -0.25     # 年齢による価格感度の調整（高齢→低感度）
    alpha_age_effect: float = -0.4       # 年齢による価格感度の調整（高齢→低感度）
    gamma_mean: float = 1.0        # 施策反応性の平均
    # gamma_std: float = 0.3         # 施策反応性の標準偏差
    gamma_std: float = 0.15          # 施策反応性の標準偏差
    # gamma_loyalty_effect: float = 0.2  # ロイヤルティによる施策反応性の調整
    gamma_loyalty_effect: float = 0.6    # ロイヤルティによる施策反応性の調整
    # beta_std: float = 0.5          # ベース選好の標準偏差（残差ノイズ）
    beta_std: float = 0.3             # ベース選好の標準偏差（残差ノイズ）
    # beta_income_effect: float = 0.3    # 所得によるベース選好の調整（高所得→高選好）
    beta_income_effect: float = 0.5      # 所得によるベース選好の調整（高所得→高選好）
    # beta_age_effect: float = 0.2       # 年齢によるベース選好の調整（高齢→高選好）
    beta_age_effect: float = 0.4         # 年齢によるベース選好の調整（高齢→高選好）
    # beta_loyalty_effect: float = 0.3   # ロイヤルティによるベース選好の調整
    beta_loyalty_effect: float = 0.5     # ロイヤルティによるベース選好の調整

    # 割引率
    rate_sale1: float = 0.05       # 日曜セール割引率
    rate_sale2: float = 0.03       # 季節セール割引率
    rate_campaign: float = 0.05    # キャンペーン割引率
    rate_coupon_base: float = 0.025    # 一般クーポン割引率
    rate_coupon_loyal: float = 0.01   # ロイヤルクーポン割引率

    # 施策頻度
    # campaign_prob: float = 0.10        # キャンペーン発生確率（日次）
    campaign_prob: float = 0.15          # キャンペーン発生確率（日次）
    campaign_max_concurrent: int = 3   # 同時開催最大店舗数
    campaign_duration: int = 3         # キャンペーン日数
    # coupon_rate_loyal: float = 0.12    # ロイヤル顧客クーポン配信確率
    coupon_rate_loyal: float = 0.18      # ロイヤル顧客クーポン配信確率
    # coupon_rate_base: float = 0.08     # 一般顧客クーポン配信確率
    coupon_rate_base: float = 0.12       # 一般顧客クーポン配信確率
    # push_rate: float = 0.15            # プッシュ通知確率
    push_rate: float = 0.25             # プッシュ通知確率
    loyalty_prob: float = 0.3          # ロイヤルティ会員確率

    # キャリブレーション
    # visit_calibration: float = 0.5 # Sigmoidの切片 i.e., 高いと行動しやすい
    visit_calibration: float = -0.3  # Sigmoidの切片（効果係数強化に伴い下方修正）
    purchase_calibration: float = -1.0 # 例: 0 → 50%, 1 → 73%, -2 → 12%

    # Gumbelノイズスケール
    # noise_scale_visit: float = 0.20      # 元0.5, Visit Utilityに加算するGumbelノイズの係数
    noise_scale_visit: float = 0.10        # Visit Utilityに加算するGumbelノイズの係数
    # noise_scale_purchase: float = 0.20   # 元0.5, Purchase Utilityに加算するGumbelノイズの係数
    noise_scale_purchase: float = 0.10     # Purchase Utilityに加算するGumbelノイズの係数

    # ラグウィンドウ
    ws: int = 4

    def __post_init__(self):
        if self.store_probs is None:
            if self.n_stores == 10:
                self.store_probs = [
                    0.20, 0.15, 0.13, 0.12, 0.10,
                    0.08, 0.07, 0.06, 0.05, 0.04,
                ]
            else:
                self.store_probs = [1.0 / self.n_stores] * self.n_stores

    def to_dict(self) -> Dict:
        """設定を辞書に変換（JSON保存用）"""
        d = {k: v for k, v in self.__dict__.items()}
        d['period_begin'] = str(d['period_begin'])
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> "SimulationConfig":
        """辞書からSimulationConfigを復元（config.jsonの読み込み用）"""
        d = d.copy()
        if 'period_begin' in d and isinstance(d['period_begin'], str):
            d['period_begin'] = datetime.date.fromisoformat(d['period_begin'])
        return cls(**d)


# ========================================
# データ生成関数
# ========================================

def generate_period_data(config: SimulationConfig) -> pd.DataFrame:
    """期間データ（日付レベルの特徴量）を生成"""
    period = pd.DataFrame({
        'date': [
            config.period_begin + datetime.timedelta(days=i)
            for i in range(config.n_days)
        ]
    })
    period['date'] = pd.to_datetime(period['date'])
    period['month'] = period['date'].dt.month
    period['dow'] = period['date'].dt.dayofweek
    period['day_of_year'] = period['date'].dt.dayofyear

    # 日曜セール
    period['sale1'] = (period['dow'] == 6).astype(int)
    # 季節セール（3,6,9,12月）
    period['sale2'] = period['month'].isin([3, 6, 9, 12]).astype(int)

    return period


def generate_consumer_data(config: SimulationConfig) -> pd.DataFrame:
    """消費者データを生成"""
    n = config.n_consumers

    consumers = pd.DataFrame({
        'user_id': range(n),
        'store_id': np.random.choice(
            config.n_stores, size=n, p=config.store_probs
        ),
        'age': np.random.randint(20, 70, size=n),
        'income': np.random.lognormal(mean=6.0, sigma=0.5, size=n),
        'loyalty': np.random.binomial(1, config.loyalty_prob, size=n),
    })

    # 潜在パラメータ（観測変数と相関）
    income_z = (
        (consumers['income'] - consumers['income'].mean())
        / consumers['income'].std()
    )
    age_z = (
        (consumers['age'] - consumers['age'].mean())
        / consumers['age'].std()
    )
    consumers['alpha'] = (
        config.alpha_mean
        + np.random.normal(0, config.alpha_std, n)
        + config.alpha_income_effect * income_z
        + config.alpha_age_effect * age_z
    )
    consumers['gamma'] = (
        config.gamma_mean
        + np.random.normal(0, config.gamma_std, n)
        + config.gamma_loyalty_effect * consumers['loyalty']
    )
    consumers['beta'] = (
        np.random.normal(0, config.beta_std, n)
        + config.beta_income_effect * income_z
        + config.beta_age_effect * age_z
        + config.beta_loyalty_effect * consumers['loyalty']
    )

    return consumers


def generate_interventions(
    config: SimulationConfig,
    consumers: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """施策変数を生成

    Returns:
        campaign: (n_stores, n_days) キャンペーンフラグ
        coupon: (n_consumers, n_days) クーポンフラグ
        push: (n_consumers, n_days) プッシュ通知フラグ
    """
    n_days = config.n_days
    n_stores = config.n_stores
    n = config.n_consumers

    # キャンペーン（突発的、同時最大制限あり）
    campaign = np.zeros((n_stores, n_days), dtype=int)
    t = 0
    while t < n_days:
        if np.random.random() < config.campaign_prob:
            n_active = np.sum(campaign[:, t])
            available = np.where(campaign[:, t] == 0)[0]
            if len(available) > 0 and n_active < config.campaign_max_concurrent:
                store = np.random.choice(available)
                end = min(t + config.campaign_duration, n_days)
                campaign[store, t:end] = 1
        t += 1

    # クーポン（ロイヤル顧客は高頻度）
    coupon = np.zeros((n, n_days), dtype=int)
    coupon_rate = np.where(
        consumers['loyalty'].values == 1,
        config.coupon_rate_loyal,
        config.coupon_rate_base,
    )
    for t in range(n_days):
        coupon[:, t] = np.random.binomial(1, coupon_rate)

    # プッシュ通知（i.i.d.）
    push = np.random.binomial(1, config.push_rate, size=(n, n_days))

    return campaign, coupon, push


def generate_prices(
    config: SimulationConfig,
    period: pd.DataFrame,
    consumers: pd.DataFrame,
    campaign: np.ndarray,
    coupon: np.ndarray,
) -> np.ndarray:
    """価格を生成（消費者×日付レベル）

    施策→価格の因果的合成:
        price = base - sale1割引 - sale2割引 - campaign割引 - coupon割引

    Returns:
        price: (n_consumers, n_days)
    """
    n = config.n_consumers
    T = config.n_days
    pb = config.price_base

    price = np.full((n, T), float(pb))
    store_ids = consumers['store_id'].values
    coupon_rate_individual = np.where(
        consumers['loyalty'].values == 1,
        config.rate_coupon_loyal,
        config.rate_coupon_base,
    )

    for t in range(T):
        price[:, t] -= period['sale1'].iloc[t] * pb * config.rate_sale1
        price[:, t] -= period['sale2'].iloc[t] * pb * config.rate_sale2
        price[:, t] -= campaign[store_ids, t] * pb * config.rate_campaign
        price[:, t] -= coupon[:, t] * pb * coupon_rate_individual

    return price


def simulate_outcomes(
    config: SimulationConfig,
    consumers: pd.DataFrame,
    campaign: np.ndarray,
    coupon: np.ndarray,
    push: np.ndarray,
    price: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """アウトカムを生成

    Visit（アプリ閲覧）と Purchase（店頭購買）を独立に生成。

    Returns:
        visit: (n_consumers, n_days)
        purchase: (n_consumers, n_days)
        histories: 時変状態変数の履歴
    """
    n = config.n_consumers
    T = config.n_days

    visit = np.zeros((n, T), dtype=int)
    purchase = np.zeros((n, T), dtype=int)

    # 累積変数
    cumulative_visits = np.zeros(n)
    cumulative_purchases = np.zeros(n)
    days_since_visit = np.ones(n) * 999
    days_since_purchase = np.ones(n) * 999

    # 履歴保存用
    hist_cum_visits = np.zeros((n, T))
    hist_cum_purchases = np.zeros((n, T))
    hist_dsv = np.zeros((n, T))
    hist_dsp = np.zeros((n, T))

    alpha = consumers['alpha'].values
    gamma = consumers['gamma'].values
    beta = consumers['beta'].values
    store_ids = consumers['store_id'].values
    loyalty = consumers['loyalty'].values

    for t in range(T):
        # 期首の状態を記録
        hist_cum_visits[:, t] = cumulative_visits
        hist_cum_purchases[:, t] = cumulative_purchases
        hist_dsv[:, t] = days_since_visit
        hist_dsp[:, t] = days_since_purchase

        # --- Visit（アプリ閲覧）---
        campaign_notify = campaign[store_ids, t]
        recency_visit = np.minimum(days_since_visit, 30) / 30.0

        U_visit = (
            config.visit_calibration
            + beta
            # + gamma * campaign_notify * 0.6
            + gamma * campaign_notify * 1.2
            # + gamma * coupon[:, t] * 0.4
            + gamma * coupon[:, t] * 0.8
            # + gamma * push[:, t] * 0.3
            + gamma * push[:, t] * 0.6
            + 0.1 * loyalty
            - 0.3 * (1.0 - recency_visit)
        )
        noise_visit = np.random.gumbel(0, 1, n)
        prob_visit = 1.0 / (1.0 + np.exp(-(U_visit + noise_visit * config.noise_scale_visit)))
        visit[:, t] = np.random.binomial(1, prob_visit)

        # --- Purchase（店頭購買）---
        price_normalized = (price[:, t] - config.price_base) / config.price_base
        recency_purchase = np.minimum(days_since_purchase, 60) / 60.0

        U_purchase = (
            config.purchase_calibration
            + beta
            - alpha * price_normalized
            # + 0.15 * loyalty
            + 0.30 * loyalty
            # - 0.5 * (1.0 - recency_purchase)
            - 0.8 * (1.0 - recency_purchase)
        )
        noise_purchase = np.random.gumbel(0, 1, n)
        prob_purchase = 1.0 / (1.0 + np.exp(-(U_purchase + noise_purchase * config.noise_scale_purchase)))
        purchase[:, t] = np.random.binomial(1, prob_purchase)

        # --- 累積更新 ---
        cumulative_visits += visit[:, t]
        cumulative_purchases += purchase[:, t]
        days_since_visit += 1
        days_since_visit[visit[:, t] == 1] = 0
        days_since_purchase += 1
        days_since_purchase[purchase[:, t] == 1] = 0

    histories = {
        'cumulative_visits': hist_cum_visits,
        'cumulative_purchases': hist_cum_purchases,
        'days_since_visit': hist_dsv,
        'days_since_purchase': hist_dsp,
    }

    return visit, purchase, histories


# ========================================
# データセット構築
# ========================================

def build_dataset(
    config: SimulationConfig,
    consumers: pd.DataFrame,
    period: pd.DataFrame,
    campaign: np.ndarray,
    coupon: np.ndarray,
    push: np.ndarray,
    price: np.ndarray,
    visit: np.ndarray,
    purchase: np.ndarray,
    histories: Dict[str, np.ndarray],
) -> pd.DataFrame:
    """パネルデータを構築（全visible unitをbinaryに）

    各行は (consumer_id, day) の組み合わせ。
    BBRBM用に全特徴量をbinary化。
    """
    n = config.n_consumers
    T = config.n_days
    ws = config.ws

    # 基本インデックス（consumer-major order）
    consumer_ids = np.repeat(np.arange(n), T)
    day_indices = np.tile(np.arange(T), n)

    df = pd.DataFrame({
        'consumer_id': consumer_ids,
        'day': day_indices,
        'date': np.tile(period['date'].values, n),
    })

    # ---- 消費者特徴量（時間不変）----

    # store_id: one-hot
    store_vals = consumers['store_id'].values
    for s in range(config.n_stores):
        df[f'store_{s}'] = np.repeat((store_vals == s).astype(int), T)

    # loyalty: binary
    df['loyalty'] = np.repeat(consumers['loyalty'].values, T)

    # age: 10歳刻みone-hot
    age_vals = consumers['age'].values
    for decade in range(20, 70, 10):
        df[f'age_{decade}s'] = np.repeat(
            ((age_vals >= decade) & (age_vals < decade + 10)).astype(int), T
        )

    # income: 5分位one-hot
    income_vals = consumers['income'].values
    boundaries = np.percentile(income_vals, [20, 40, 60, 80])
    income_bin = np.digitize(income_vals, boundaries)  # 0,1,2,3,4
    for q in range(5):
        df[f'income_q{q + 1}'] = np.repeat(
            (income_bin == q).astype(int), T
        )

    # ---- 潜在パラメータ（ground truth, 評価用・visible unitではない）----
    for col in ['alpha', 'gamma', 'beta']:
        df[f'true_{col}'] = np.repeat(consumers[col].values, T)

    # ---- 期間特徴量 ----

    # month: one-hot (1-12)
    month_vals = period['month'].values
    for m in range(1, 13):
        df[f'month_{m}'] = np.tile((month_vals == m).astype(int), n)

    # dow: one-hot (0=Mon, ..., 6=Sun)
    dow_vals = period['dow'].values
    for d in range(7):
        df[f'dow_{d}'] = np.tile((dow_vals == d).astype(int), n)

    # sale1, sale2: そのままbinary
    df['sale1'] = np.tile(period['sale1'].values, n)
    df['sale2'] = np.tile(period['sale2'].values, n)

    # ---- 施策変数 ----
    store_ids = consumers['store_id'].values
    df['campaign'] = campaign[store_ids].flatten()
    df['coupon'] = coupon.flatten()
    df['push'] = push.flatten()

    # ---- アウトカム ----
    df['visit'] = visit.flatten()
    df['purchase'] = purchase.flatten()

    # ---- 時変状態変数（binary化）----

    # days_since_visit → 7日以内/14日以内/30日以内 ダミー
    dsv = histories['days_since_visit']
    df['visited_within_7d'] = (dsv <= 7).astype(int).flatten()
    df['visited_within_14d'] = (dsv <= 14).astype(int).flatten()
    df['visited_within_30d'] = (dsv <= 30).astype(int).flatten()

    # days_since_purchase → 7日以内/14日以内/30日以内 ダミー
    dsp = histories['days_since_purchase']
    df['purchased_within_7d'] = (dsp <= 7).astype(int).flatten()
    df['purchased_within_14d'] = (dsp <= 14).astype(int).flatten()
    df['purchased_within_30d'] = (dsp <= 30).astype(int).flatten()

    # cumulative_visits → 5回以上/10回以上/30回以上 ダミー
    cv = histories['cumulative_visits']
    df['cum_visits_5plus'] = (cv >= 5).astype(int).flatten()
    df['cum_visits_10plus'] = (cv >= 10).astype(int).flatten()
    df['cum_visits_30plus'] = (cv >= 30).astype(int).flatten()

    # cumulative_purchases → 5回以上/10回以上/30回以上 ダミー
    cp = histories['cumulative_purchases']
    df['cum_purchases_5plus'] = (cp >= 5).astype(int).flatten()
    df['cum_purchases_10plus'] = (cp >= 10).astype(int).flatten()
    df['cum_purchases_30plus'] = (cp >= 30).astype(int).flatten()

    # ---- ラグ特徴量（binary変数のみ）----
    lag_targets = {
        'visit': (visit, 0),
        'purchase': (purchase, 0),
        'coupon': (coupon, 0),
        'campaign': (campaign[store_ids], 0),
        'push': (push, 0),
    }

    for lag in range(1, ws + 1):
        for name, (arr, fill_val) in lag_targets.items():
            shifted = np.full_like(arr, fill_val, dtype=float)
            if lag < T:
                shifted[:, lag:] = arr[:, :-lag]
            df[f'{name}_lag{lag}'] = shifted.flatten().astype(int)

    return df


# DBM投入時のカラム分類
ID_COLS = {'consumer_id', 'day', 'date'}
GROUND_TRUTH_COLS = {'true_alpha', 'true_gamma', 'true_beta'}
OUTCOME_COLS = {'visit', 'purchase'}
ACTION_COLS = {'sale1', 'sale2', 'campaign', 'coupon', 'push'}
EXCLUDE_COLS = ID_COLS | GROUND_TRUTH_COLS | OUTCOME_COLS | ACTION_COLS


def get_visible_cols(df: pd.DataFrame) -> list:
    """DataFrameからBBRBMのvisible unitカラム名リストを返す"""
    return [c for c in df.columns if c not in EXCLUDE_COLS]


def get_action_cols(df: pd.DataFrame) -> list:
    """DataFrameからアクション変数のカラム名リストを返す（Adapter入力用）"""
    return [c for c in df.columns if c in ACTION_COLS]


# ========================================
# メイン関数
# ========================================

def generate_simulation_data(
    config: Optional[SimulationConfig] = None,
    output_dir: Optional[str] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """シミュレーションデータを生成するメイン関数

    Args:
        config: シミュレーション設定。Noneの場合はデフォルト値を使用。
        output_dir: 出力先ディレクトリ。Noneの場合は保存しない。
        verbose: 進捗・統計情報を表示するか。

    Returns:
        パネルデータの DataFrame
    """
    if config is None:
        config = SimulationConfig()

    np.random.seed(config.seed)

    if verbose:
        print("Generating simulation data...")
        print(f"  Consumers: {config.n_consumers}")
        print(f"  Days: {config.n_days}")
        print(f"  Stores: {config.n_stores}")
        print(f"  Lag window: {config.ws}")

    # 1. 期間データ
    if verbose:
        print("  Generating period data...")
    period = generate_period_data(config)

    # 2. 消費者データ
    if verbose:
        print("  Generating consumer data...")
    consumers = generate_consumer_data(config)

    # 3. 施策変数
    if verbose:
        print("  Generating interventions...")
    campaign, coupon, push = generate_interventions(config, consumers)

    # 4. 価格
    if verbose:
        print("  Generating prices...")
    price = generate_prices(config, period, consumers, campaign, coupon)

    # 5. アウトカム
    if verbose:
        print("  Simulating outcomes...")
    visit, purchase, histories = simulate_outcomes(
        config, consumers, campaign, coupon, push, price
    )

    # 6. データセット構築
    if verbose:
        print("  Building dataset...")
    df = build_dataset(
        config, consumers, period, campaign, coupon, push, price,
        visit, purchase, histories
    )

    # 統計サマリ
    if verbose:
        print(f"\n--- Simulation Statistics ---")
        print(f"Total records: {len(df):,}")
        print(f"Visit rate: {df['visit'].mean():.4f}")
        print(f"Purchase rate: {df['purchase'].mean():.4f}")
        print(f"Campaign rate: {df['campaign'].mean():.4f}")
        print(f"Coupon rate: {df['coupon'].mean():.4f}")
        print(f"Push rate: {df['push'].mean():.4f}")
        print(f"\nHeterogeneity:")
        print(f"  corr(alpha, income):  {consumers['alpha'].corr(consumers['income']):.3f}")
        print(f"  corr(alpha, age):     {consumers['alpha'].corr(consumers['age']):.3f}")
        print(f"  corr(gamma, loyalty): {consumers['gamma'].corr(consumers['loyalty']):.3f}")
        print(f"  corr(beta, income):   {consumers['beta'].corr(consumers['income']):.3f}")
        print(f"  corr(beta, age):      {consumers['beta'].corr(consumers['age']):.3f}")
        print(f"  corr(beta, loyalty):  {consumers['beta'].corr(consumers['loyalty']):.3f}")

    # 保存
    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        df.to_csv(output_path / 'simulation_data.csv', index=False)

        with open(output_path / 'config.json', 'w') as f:
            json.dump(config.to_dict(), f, indent=2, ensure_ascii=False)

        if verbose:
            print(f"\nSaved to: {output_path}")

    return df


def create_train_val_test_split(
    df: pd.DataFrame,
    test_days: int = 60,
    val_days: int = 0,
    output_dir: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """時間ベースの train/val/test 分割

    時系列の末尾から test_days, val_days を順に切り出す。
    val_days=0 の場合は val_df は空の DataFrame を返す（後方互換）。

    Timeline: [--- train ---|--- val ---|--- test ---]

    Args:
        df: パネルデータ
        test_days: テスト期間の日数
        val_days: バリデーション期間の日数（0で無効）
        output_dir: 保存先ディレクトリ

    Returns:
        train_df, val_df, test_df
    """
    max_day = df['day'].max()
    total_hold_out = test_days + val_days
    if total_hold_out >= max_day + 1:
        raise ValueError(
            f"test_days ({test_days}) + val_days ({val_days}) = {total_hold_out} "
            f">= total days ({max_day + 1}). Reduce test_days or val_days."
        )

    test_cutoff = max_day - test_days + 1
    val_cutoff = test_cutoff - val_days

    train_df = df[df['day'] < val_cutoff].copy()
    val_df = df[(df['day'] >= val_cutoff) & (df['day'] < test_cutoff)].copy()
    test_df = df[df['day'] >= test_cutoff].copy()

    print(f"Train: days 0-{val_cutoff - 1}, {len(train_df):,} records")
    if val_days > 0:
        print(f"Val:   days {val_cutoff}-{test_cutoff - 1}, {len(val_df):,} records")
    print(f"Test:  days {test_cutoff}-{max_day}, {len(test_df):,} records")

    if output_dir is not None:
        output_path = Path(output_dir)
        train_df.to_csv(output_path / 'train.csv', index=False)
        if val_days > 0:
            val_df.to_csv(output_path / 'val.csv', index=False)
        test_df.to_csv(output_path / 'test.csv', index=False)

    return train_df, val_df, test_df


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate marketing simulation data'
    )
    parser.add_argument('--n_consumers', type=int, default=1024)
    parser.add_argument('--n_days', type=int, default=365)
    parser.add_argument('--n_stores', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--ws', type=int, default=4)
    parser.add_argument('--output_dir', type=str, default='./data/simulation')
    parser.add_argument('--test_days', type=int, default=20)

    args = parser.parse_args()

    config = SimulationConfig(
        n_consumers=args.n_consumers,
        n_days=args.n_days,
        n_stores=args.n_stores,
        seed=args.seed,
        ws=args.ws,
    )

    df = generate_simulation_data(config, output_dir=args.output_dir)

    print("\n--- Train/Test Split ---")
    train_df, test_df = create_train_test_split(
        df, test_days=args.test_days, output_dir=args.output_dir
    )
