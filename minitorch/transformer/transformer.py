from __future__ import annotations

import numpy as np

from minitorch.tensor.tensor import Tensor
from minitorch.activations.activations import GELU
from minitorch.nn.layers import Linear
from minitorch.attention.attention import MultiHeadAttention
from minitorch.embendding.embed import EmbeddingLayer
#ghhjkkjgf


def create_causal_mask(seq_len: int) -> Tensor:
    """
    Create a causal mask (autoregressive mask).
    
    This create the causal mask to make sure that tokens i
    only communicates to token j where j<i.
    Essential for autoregressive GPT models.

    Args:
        seq_len (int): Length of the sequence

    Returns:
        Tensor: Tensor of shape (1, seq_len, seq_len) with:
        - 1.0 for positions that CAN be attended to (lower triangle)
        - 0.0 for positions that CANNOT be attended to (upper triangle)
    """
    mask = np.tril(np.ones(shape=(seq_len, seq_len), dtype= np.float32))
    return Tensor(mask[np.newaxis, :, :])