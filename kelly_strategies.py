"""
kelly_binary_epl.py

Binary version tailored to the Yeung-style EPL dataset with:

- data_tree-style columns (LR_* predicted stats, GAP/AVG features, FIFA ratings)
- B365_home_win_Prob (Bet365 home win probability)
- home_result (1 = home win, 0 = not home win)

We:
1. Train a global XGBoost classifier to predict home_result.
2. Derive model home-win probabilities (p_home).
3. Approximate home odds from B365_home_win_Prob.
4. Compute Kelly fraction for betting ONLY on home wins.
5. Classify matches by Kelly index into easy/medium/hard.
6. Train per-Kelly-group models and compare to the global model.
7. Simulate Kelly betting strategies.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict, Tuple

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay

# ---------------------------------------------------------------------
# CONFIG – EDIT DATA_PATH TO POINT TO YOUR MERGED TREE/LR DATASET
# ---------------------------------------------------------------------

DATA_PATH = "data_tree.xlsx"  # or "your_merged_dataset.csv"
TARGET_COL = "home_result"    # 1 = home win, 0 = not home win

# Features taken from your description of the LR+Tree dataset
FEATURE_COLS = [

    # LR predicted stats (from LR model)
    "LR_h_shoton", "LR_h_shotoff", "LR_h_shot", "LR_h_cross",
    "LR_a_shoton", "LR_a_shotoff", "LR_a_shot", "LR_a_cross",

    # Raw match stats
    "home_shoton", "away_shoton",
    "home_shotoff", "away_shotoff",
    "home_shot", "away_shot",
    "home_cross_sum", "away_cross_sum",

    # GAP features
    "GAP_H_Shoton", "GAP_A_Shoton",
    "GAP_H_Shotoff", "GAP_A_Shotoff",
    "GAP_H_Shot", "GAP_A_Shot",
    "GAP_H_Cross", "GAP_A_Cross",

    # AVG features
    "AVG_H_Shoton", "AVG_A_Shoton",
    "AVG_H_Shotoff", "AVG_A_Shotoff",
    "AVG_H_Shot", "AVG_A_Shot",
    "AVG_H_Cross", "AVG_A_Cross",

    # FIFA ratings (HOME)
    "CB_POW_H", "CB_MEN_H", "CB_SKI_H", "CB_MOV_H", "CB_ATT_H", "CB_DEF_H",
    "CM_POW_H", "CM_MEN_H", "CM_SKI_H", "CM_MOV_H", "CM_ATT_H", "CM_DEF_H",
    "GK_POW_H", "GK_MEN_H", "GK_MOV_H", "GK_GOK_H",
    "ST_POW_H", "ST_MEN_H", "ST_SKI_H", "ST_MOV_H", "ST_ATT_H", "ST_DEF_H",

    # FIFA ratings (AWAY)
    "CB_POW_A", "CB_MEN_A", "CB_SKI_A", "CB_MOV_A", "CB_ATT_A", "CB_DEF_A",
    "CM_POW_A", "CM_MEN_A", "CM_SKI_A", "CM_MOV_A", "CM_ATT_A", "CM_DEF_A",
    "GK_POW_A", "GK_MEN_A", "GK_MOV_A", "GK_GOK_A",
    "ST_POW_A", "ST_MEN_A", "ST_SKI_A", "ST_MOV_A", "ST_ATT_A", "ST_DEF_A",

    # Bookmaker probability (used as feature AND for synthetic odds)
    "B365_home_win_Prob",
]


# ---------------------------------------------------------------------
# Kelly utilities – BINARY HOME-WIN VERSION
# ---------------------------------------------------------------------

def synthesize_home_odds_from_prob(df: pd.DataFrame, prob_col: str) -> pd.DataFrame:
    """
    Approximate decimal odds for home win from Bet365 implied probability:

        odds ≈ 1 / p_book

    NOTE: This ignores the full overround structure; it's a simplification
    but enough for your experimental comparison.
    """
    df["home_odds_synth"] = 1.0 / df[prob_col]
    return df


def compute_kelly_fraction_binary(p: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Binary Kelly for betting on "home win":

        f* = (b*p - q) / b

    where:
    - p is model probability of home win
    - q = 1 - p
    - b = decimal_odds - 1
    """
    q = 1.0 - p
    return (b * p - q) / b


def compute_kelly_index_binary(df: pd.DataFrame,
                               model_prob_col: str = "p_home",
                               odds_col: str = "home_odds_synth") -> pd.DataFrame:
    """
    Compute Kelly fraction for betting on home win only:

    - uses model probability p_home
    - uses synthetic odds from B365_home_win_Prob

    Adds:
        - 'kelly_fraction_home'
        - 'kelly_index' (same as kelly_fraction_home)
    """
    p = df[model_prob_col].values
    b = df[odds_col].values - 1.0
    kf = compute_kelly_fraction_binary(p, b)

    df["kelly_fraction_home"] = kf
    df["kelly_index"] = kf  # you can cap negatives later if desired
    return df


def classify_match_difficulty(
    df: pd.DataFrame,
    index_col: str = "kelly_index",
    q_easy: float = 0.66,
    q_hard: float = 0.33,
) -> pd.DataFrame:
    """
    Label matches as 'easy', 'medium', 'hard' based on Kelly index quantiles.

    easy  : kelly_index >= q_easy quantile
    medium: between q_hard and q_easy
    hard  : kelly_index < q_hard quantile
    """
    q_hi = df[index_col].quantile(q_easy)
    q_lo = df[index_col].quantile(q_hard)

    def label(val: float) -> str:
        if val >= q_hi:
            return "easy"
        elif val < q_lo:
            return "hard"
        else:
            return "medium"

    df["kelly_group"] = df[index_col].apply(label)
    return df


# ---------------------------------------------------------------------
# Betting backtest (home-win only)
# ---------------------------------------------------------------------

@dataclass
class BetResult:
    final_bankroll: float
    roi: float
    n_bets: int
    hit_rate: float
    max_drawdown: float


def simulate_home_kelly_strategy(
    df: pd.DataFrame,
    stake_fraction: float = 0.25,   # fraction of full Kelly
    min_kelly: float = 0.0,         # bet only if kelly_fraction_home >= min_kelly
    initial_bankroll: float = 1000.0,
    group_filter: str = None,       # 'easy', 'medium', 'hard', or None
) -> BetResult:
    """
    Simulate betting ONLY on home wins:

    - For each match, if kelly_fraction_home >= min_kelly and (optional) group matches,
      stake = stake_fraction * kelly_fraction_home * bankroll.
    - If home_result == 1, win stake * (odds - 1).
      Else, lose stake.
    """
    bankroll = initial_bankroll
    peak_bankroll = initial_bankroll
    n_bets = 0
    n_hits = 0

    history = []

    for _, row in df.iterrows():
        if group_filter is not None and row.get("kelly_group") != group_filter:
            continue

        kf = row["kelly_fraction_home"]
        if kf < min_kelly:
            continue

        odds = row["home_odds_synth"]
        b = odds - 1.0

        stake = stake_fraction * kf * bankroll
        if stake <= 0:
            continue

        n_bets += 1
        actual = row[TARGET_COL]

        if actual == 1:
            profit = stake * b
            bankroll += profit
            n_hits += 1
        else:
            bankroll -= stake

        peak_bankroll = max(peak_bankroll, bankroll)
        history.append(bankroll)

    roi = (bankroll - initial_bankroll) / initial_bankroll if initial_bankroll > 0 else 0.0
    hit_rate = n_hits / n_bets if n_bets > 0 else 0.0

    max_drawdown = 0.0
    if history:
        peak = initial_bankroll
        max_dd = 0.0
        for bkr in history:
            peak = max(peak, bkr)
            dd = (peak - bkr) / peak
            max_dd = max(max_dd, dd)
        max_drawdown = max_dd

    return BetResult(
        final_bankroll=bankroll,
        roi=roi,
        n_bets=n_bets,
        hit_rate=hit_rate,
        max_drawdown=max_drawdown,
    )


# ---------------------------------------------------------------------
# Modeling: global + per-group XGBoost (binary)
# ---------------------------------------------------------------------

def train_xgb_classifier_binary(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = TARGET_COL,
    random_state: int = 42,
) -> Tuple[XGBClassifier, Dict[str, float], Dict[str, np.ndarray]]:
    """
    Train a binary XGBoost classifier for home_result.
    Returns model + metrics on a holdout set.
    """
    X = df[feature_cols].values
    y = df[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state, stratify=y
    )

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=random_state,
    )

    model.fit(X_train, y_train)

    y_pred = (model.predict_proba(X_test)[:, 1] >= 0.5).astype(int)
    y_proba = model.predict_proba(X_test)[:, 1]

    f1 = f1_score(y_test, y_pred, average="binary")
    roc = roc_auc_score(y_test, y_proba)

    metrics = {"f1": f1, "roc_auc": roc}
    eval_payload = {
        "y_test": y_test,
        "y_pred": y_pred,
        "y_proba": y_proba,
    }
    return model, metrics, eval_payload


def train_group_models_binary(
    df: pd.DataFrame,
    group_col: str = "kelly_group",
    feature_cols: List[str] = None,
    target_col: str = TARGET_COL,
    random_state: int = 42,
) -> Dict[str, Tuple[XGBClassifier, Dict[str, float], Dict[str, np.ndarray]]]:
    """
    Train separate binary models for each Kelly group: easy/medium/hard.

    Returns dict[group] = (model, metrics, eval_payload).
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS

    models = {}
    for group in sorted(df[group_col].unique()):
        subset = df[df[group_col] == group]
        if len(subset) < 50:
            print(f"[INFO] Skipping group '{group}' (too few samples: {len(subset)})")
            continue
        model, metrics, eval_payload = train_xgb_classifier_binary(
            subset,
            feature_cols=feature_cols,
            target_col=target_col,
            random_state=random_state,
        )
        models[group] = (model, metrics, eval_payload)
    return models


# ---------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------

def main():
    # 1. Load data
    if DATA_PATH.endswith(".xlsx"):
        df = pd.read_excel(DATA_PATH)
    else:
        df = pd.read_csv(DATA_PATH)

    # Keep only rows with all required columns
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL, "B365_home_win_Prob"]).copy()

    # 2. Train global model
    print("=== Training global XGBoost model (binary home win) ===")
    global_model, global_metrics, global_eval = train_xgb_classifier_binary(df, FEATURE_COLS)
    print("Global model metrics:", global_metrics)

    cm = confusion_matrix(global_eval["y_test"], global_eval["y_pred"])
    disp = ConfusionMatrixDisplay(cm, display_labels=["Not home win", "Home win"])
    disp.plot(colorbar=False, cmap="Blues")
    plt.title("Global model confusion matrix (holdout)")
    plt.tight_layout()
    plt.show()

    # 3. Get model probabilities for home win
    df["p_home"] = global_model.predict_proba(df[FEATURE_COLS].values)[:, 1]

    # 4. Build synthetic home odds from Bet365 probability
    df = synthesize_home_odds_from_prob(df, prob_col="B365_home_win_Prob")

    # 5. Compute Kelly index + difficulty groups
    df = compute_kelly_index_binary(df, model_prob_col="p_home", odds_col="home_odds_synth")
    df = classify_match_difficulty(df, index_col="kelly_index")

    # 5b. Visualize Kelly index distribution + group sizes
    kelly_series = df["kelly_index"].dropna()
    q_lo, q_hi = kelly_series.quantile([0.33, 0.66])

    plt.figure(figsize=(9, 4))
    plt.hist(kelly_series, bins=40, color="steelblue", alpha=0.7, edgecolor="white")
    plt.axvline(q_lo, color="orange", linestyle="--", label="33% quantile (hard/medium)")
    plt.axvline(q_hi, color="green", linestyle="--", label="66% quantile (medium/easy)")
    plt.title("Distribution of Kelly Index (Home bets)")
    plt.xlabel("Kelly fraction")
    plt.ylabel("Match count")
    plt.legend()
    plt.tight_layout()
    plt.show()

    group_counts = (
        df["kelly_group"]
            .value_counts()
            .reindex(["easy", "medium", "hard"])
            .fillna(0)
    )
    plt.figure(figsize=(6, 4))
    ax = group_counts.plot(kind="bar", color=["green", "goldenrod", "tomato"])
    plt.ylabel("Match count")
    plt.title("Kelly group counts")
    plt.xticks(rotation=0)
    for idx, value in enumerate(group_counts):
        ax.text(
            idx,
            value + max(1, group_counts.max()) * 0.01,
            f"{int(value)}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    plt.tight_layout()
    plt.show()

    # 6. Strategy A: global model + Kelly filter betting
    print("\n=== Strategy A: Kelly-based betting using global model ===")

    result_easy = simulate_home_kelly_strategy(
        df,
        stake_fraction=0.25,
        min_kelly=0.0,
        initial_bankroll=1000.0,
        group_filter="easy",
    )
    print("Easy matches only:")
    print(result_easy)

    result_all = simulate_home_kelly_strategy(
        df,
        stake_fraction=0.25,
        min_kelly=0.0,
        initial_bankroll=1000.0,
        group_filter=None,
    )
    print("\nAll matches (no Kelly group filter):")
    print(result_all)

    # 7. Strategy B: Per-Kelly-group models vs global model
    print("\n=== Strategy B: Per-group models vs global model ===")
    group_models = train_group_models_binary(df, feature_cols=FEATURE_COLS)
    for group, (_, metrics, _) in group_models.items():
        print(f"Group '{group}' model metrics:", metrics)

    # (Optional) Save with Kelly fields for analysis in your paper/project
    df.to_csv("matches_with_kelly_output.csv", index=False)
    print("\nSaved enriched dataset to matches_with_kelly_output.csv")



if __name__ == "__main__":
    main()
