# src/evaluation_utils.py

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


def build_auc_summary(results_aucs: Dict[str, List[float]]) -> pd.DataFrame:
    """
    Turn the 'results["AUCs"]' dict into a nice summary DataFrame
    with mean and std, sorted by mean AUC.
    """
    summary_data = []
    for name, auc_list in results_aucs.items():
        mean_auc = float(np.mean(auc_list))
        std_auc = float(np.std(auc_list))
        summary_data.append(
            {
                "Experiment": name.replace("_", " ").title(),
                "Mean AUC": mean_auc,
                "Std Dev (±)": std_auc,
            }
        )

    df = (
        pd.DataFrame(summary_data)
        .set_index("Experiment")
        .sort_values(by="Mean AUC", ascending=False)
    )
    return df


def _cr_store_to_df(store: dict, label: str) -> pd.DataFrame:
    """
    Helper to aggregate macro/weighted classification_report across folds.
    Directly based on your original `_cr_store_to_df`.
    """
    rows = []
    for name, arr in store.items():
        A = np.asarray(arr, dtype=float)  # shape (n_folds, 3)
        mu = A.mean(axis=0)
        sd = A.std(axis=0)
        rows.append(
            [
                name,
                mu[0],
                sd[0],
                mu[1],
                sd[1],
                mu[2],
                sd[2],
            ]
        )
    return pd.DataFrame(
        rows,
        columns=[
            "Model",
            f"{label} Precision (mean)",
            f"{label} Precision (std)",
            f"{label} Recall (mean)",
            f"{label} Recall (std)",
            f"{label} F1 (mean)",
            f"{label} F1 (std)",
        ],
    ).sort_values("Model").reset_index(drop=True)


def build_cr_tables(cr_macro: dict, cr_weighted: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Given the `cr_macro` and `cr_weighted` dicts collected over folds,
    return (macro_df, weighted_df).
    """
    macro_df = _cr_store_to_df(cr_macro, "Macro")
    weighted_df = _cr_store_to_df(cr_weighted, "Weighted")
    return macro_df, weighted_df
