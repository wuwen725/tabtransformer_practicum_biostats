# src/plot_utils.py

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.axes import Axes
from sklearn.metrics import auc

OUT_DIR = Path("output")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def savefig_in_outdir(filename: str, dpi: int = 300) -> Path:
    """
    Save current matplotlib figure into OUT_DIR with tight layout and return path.
    """
    fp = OUT_DIR / filename
    plt.savefig(fp, dpi=dpi, bbox_inches="tight")
    print(f"[saved] {fp}")
    return fp


def plot_mean_roc_curves_with_std(
    mean_fpr: np.ndarray,
    results_tprs: Dict[str, List[np.ndarray]],
    results_aucs: Dict[str, List[float]],
    model_keys: List[str],
    title: str,
    save_name: str | None = None,
) -> None:
    """
    Plot multiple mean ROC curves with shaded std bands.
    This is almost a direct抽取 of your original function.
    """
    plt.figure(figsize=(10, 8))
    colors = sns.color_palette("Set2", n_colors=len(model_keys))
    linestyles = ["-", "--", ":", "-."]

    for i, key in enumerate(model_keys):
        if key not in results_tprs or not results_tprs[key]:
            print(f"Warning: No results found for '{key}'. Skipping this curve.")
            continue

        tpr_array = np.array(results_tprs[key])
        mean_tprs = tpr_array.mean(axis=0)
        mean_tprs[-1] = 1.0

        mean_auc = auc(mean_fpr, mean_tprs)
        std_auc = float(np.std(results_aucs[key]))

        std_tpr = tpr_array.std(axis=0)
        tprs_upper = np.minimum(mean_tprs + std_tpr, 1)
        tprs_lower = np.maximum(mean_tprs - std_tpr, 0)

        plt.plot(
            mean_fpr,
            mean_tprs,
            color=colors[i],
            linestyle=linestyles[i % len(linestyles)],
            lw=2.5,
            label=f"{key.replace('_', ' ').title()} (AUC = {mean_auc:.3f} ± {std_auc:.3f})",
        )
        plt.fill_between(
            mean_fpr,
            tprs_lower,
            tprs_upper,
            color=colors[i],
            alpha=0.15,
        )

    plt.plot([0, 1], [0, 1], color="black", lw=1, linestyle="--", label="Chance")
    plt.title(title, fontsize=16)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right", title="Model (Mean AUC ± Std Dev)")
    plt.grid(True)
    sns.despine()

    if save_name is not None:
        savefig_in_outdir(save_name)

    plt.show()
