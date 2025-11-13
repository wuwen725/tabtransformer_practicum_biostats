# src/train_utils.py

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from .model_utils import build_tabtransformer


def train_tabtransformer_in_cv(
    X_cat_train: torch.Tensor,
    X_cont_train: torch.Tensor,
    y_train: Sequence[int],
    X_cat_test: torch.Tensor,
    X_cont_test: torch.Tensor,
    cat_cardinalities: list[int],
    n_epochs: int = 50,
    lr: float = 1e-4,
) -> np.ndarray:
    """
    Train a TabTransformer for a single CV fold and return predicted probabilities
    on the test set.
    
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)

    model = build_tabtransformer(
        cat_cardinalities=cat_cardinalities,
        num_continuous=X_cont_train.shape[1],
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr)

    pos = float(np.sum(y_train))
    neg = float(len(y_train) - pos)
    w = torch.tensor(
        [neg / (pos + neg), pos / (pos + neg)],
        dtype=torch.float32,
    )
    criterion = nn.CrossEntropyLoss(weight=w.to(device))

    for _ in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_cat_train.to(device), X_cont_train.to(device))
        loss = criterion(output, y_train_tensor.to(device))
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        test_output = model(X_cat_test.to(device), X_cont_test.to(device))
        y_pred_proba = torch.softmax(test_output, dim=1).cpu().numpy()[:, 1]

    return y_pred_proba
