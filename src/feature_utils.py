# src/feature_utils.py

from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.preprocessing import StandardScaler

from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats


# =========================
# Global helpers (non-CV)
# =========================

DEFAULT_CLINICAL_COLS = [
    "age",
    "age_of_menarche",
    "menopausal_status",
    "brca1",
    "brca2",
    "blood_relatives_cancer",
    "race",
    "hispanic",
    "bmi",
    "ever_pregnant",
    "tyrer_cuzick_score",
]


def split_clinical_and_expression(
    X: pd.DataFrame,
    clinical_cols: Iterable[str] = DEFAULT_CLINICAL_COLS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Given full feature matrix X, return (clinical_df, gene_counts_df).
    """
    cat_cols_in_df = [c for c in clinical_cols if c in X.columns]
    clinical_df = X[cat_cols_in_df].copy()
    gene_df = X.drop(columns=cat_cols_in_df)
    return clinical_df, gene_df


def log_transform_counts(counts_df: pd.DataFrame) -> pd.DataFrame:
    """log1p transform for PCA / heatmap etc."""
    return np.log1p(counts_df)


def run_deseq2(counts_df: pd.DataFrame, y: pd.Series) -> tuple[DeseqDataSet, pd.DataFrame]:
    """
    Run DESeq2 (via pydeseq2) on counts_df with binary label y (0/1).
    Returns (dds, results_df).
    """
    samples_df = pd.DataFrame({"condition": y})
    dds = DeseqDataSet(counts=counts_df, metadata=samples_df, design_factors="condition")
    dds.deseq2()
    stat_res = DeseqStats(dds, contrast=["condition", 1, 0])
    stat_res.summary()
    results_df = stat_res.results_df
    return dds, results_df


def filter_significant_genes(
    results_df: pd.DataFrame,
    padj_thresh: float = 0.05,
    log2fc_thresh: float = 1.0,
) -> list[str]:
    """
    Select significant genes based on padj and |log2FC|.
    """
    mask = (results_df["padj"] < padj_thresh) & (
        results_df["log2FoldChange"].abs() > log2fc_thresh
    )
    return results_df.index[mask].tolist()


# =========================
# Fold-level feature engineering
# (directly adapted from process_fold_features)
# =========================

def _clean_numeric_cols(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["age", "age_of_menarche", "bmi"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def process_fold_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    known_cat_cols: Iterable[str] = DEFAULT_CLINICAL_COLS,
    padj_thresh: float = 0.05,
    log2fc_thresh: float = 1.0,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    list[str],
    pd.DataFrame,
    pd.DataFrame,
    list[str],
]:

    cat_cols = [col for col in known_cat_cols if col in X_train.columns]
    X_train_cat = X_train[cat_cols].copy()
    X_test_cat = X_test[cat_cols].copy()

    if "tyrer_cuzick_score" in X_train_cat.columns:
        for df in (X_train_cat, X_test_cat):
            df.drop(columns=["tyrer_cuzick_score"], inplace=True)

    X_train_cat = _clean_numeric_cols(X_train_cat)
    X_test_cat = _clean_numeric_cols(X_test_cat)

    imputation_values = {
        col: (
            X_train_cat[col].median()
            if pd.api.types.is_numeric_dtype(X_train_cat[col])
            else X_train_cat[col].mode()[0]
        )
        for col in X_train_cat.columns
    }
    for col, value in imputation_values.items():
        X_train_cat[col].fillna(value, inplace=True)
        X_test_cat[col].fillna(value, inplace=True)

    bmi_bins = [0, 18.5, 25, 30, float("inf")]
    bmi_labels = ["Underweight", "Normal", "Overweight", "Obese"]

    _, menarche_bins = pd.qcut(
        X_train_cat["age_of_menarche"], q=3, retbins=True, duplicates="drop"
    )
    menarche_labels = ["Early", "Normal", "Late"]
    if len(menarche_bins) - 1 != len(menarche_labels):
        menarche_labels = [f"g{k+1}" for k in range(len(menarche_bins) - 1)]

    _, age_bins = pd.qcut(
        X_train_cat["age"], q=5, retbins=True, duplicates="drop"
    )
    age_labels = [f"a{k+1}" for k in range(len(age_bins) - 1)]

    for df in (X_train_cat, X_test_cat):
        df["age_category"] = pd.cut(
            df["age"], bins=age_bins, labels=age_labels, include_lowest=True
        )
        df["bmi_category"] = pd.cut(
            df["bmi"], bins=bmi_bins, labels=bmi_labels, right=False
        )
        df["menarche_category"] = pd.cut(
            df["age_of_menarche"],
            bins=menarche_bins,
            labels=menarche_labels,
            include_lowest=True,
        )
        df.drop(columns=["age", "bmi", "age_of_menarche"], inplace=True)

    candidate_for_chi2 = [
        c
        for c in X_train_cat.columns
        if c.endswith("_category")
        or c
        in [
            "menopausal_status",
            "brca1",
            "brca2",
            "blood_relatives_cancer",
            "race",
            "hispanic",
            "ever_pregnant",
        ]
    ]
    selected_categorical_features = [
        col
        for col in candidate_for_chi2
        if chi2_contingency(
            pd.crosstab(X_train_cat[col].astype("category"), y_train)
        )[1]
        < 0.05
    ]

    gene_cols = X_train.columns.drop(cat_cols)
    X_train_cont_raw = X_train[gene_cols].copy()
    X_test_cont_raw = X_test[gene_cols].copy()

    X_train_cont_raw = X_train_cont_raw.T.groupby(level=0).sum().T
    all_gene_columns = X_train_cont_raw.columns
    X_test_cont_raw = (
        X_test_cont_raw.T.groupby(level=0).sum().T.reindex(columns=all_gene_columns, fill_value=0)
    )

    samples_df = pd.DataFrame({"condition": y_train})
    dds = DeseqDataSet(
        counts=X_train_cont_raw, metadata=samples_df, design_factors="condition"
    )
    dds.deseq2()
    stat_res = DeseqStats(dds, contrast=["condition", 1, 0])
    stat_res.summary()
    results_df = stat_res.results_df
    significant_genes = filter_significant_genes(
        results_df, padj_thresh=padj_thresh, log2fc_thresh=log2fc_thresh
    )

    return (
        X_train_cat,
        X_test_cat,
        selected_categorical_features,
        X_train_cont_raw,
        X_test_cont_raw,
        significant_genes,
    )
