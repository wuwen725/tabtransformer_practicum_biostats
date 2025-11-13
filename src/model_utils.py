# src/model_utils.py

from typing import List

from tab_transformer_pytorch import TabTransformer
import torch.nn as nn


def build_tabtransformer(
    cat_cardinalities: List[int],
    num_continuous: int,
    dim: int = 32,
    dim_out: int = 2,
    depth: int = 6,
    heads: int = 8,
) -> nn.Module:
    """
    Factory function to build a TabTransformer model, matching the
    configuration used in the original cross-validation code.
    """
    model = TabTransformer(
        categories=cat_cardinalities,
        num_continuous=num_continuous,
        dim=dim,
        dim_out=dim_out,
        depth=depth,
        heads=heads,
    )
    return model
