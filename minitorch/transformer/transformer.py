from __future__ import annotations

import numpy as np

from typing import Dict
from minitorch.tensor.tensor import Tensor
from minitorch.activations.activations import GELU
from minitorch.nn.layers import Linear, LayerNormalization
from minitorch.attention.attention import MultiHeadAttention
from minitorch.embendding.embed import EmbeddingLayer
#ghhjkkjgf

class TransformerBlock:
    def __init__(self):
        pass

class GPT:
    """
    Complete Generative Pre-trained Transformer Model.
    
    Combines embeddings, positional enncoding, multiple transformer blocks
    and a langiage modeling head for text generation.
    
    """
    def __init__(self, config: Dict):
        """
        Initialize the GPT model
        
        APPROACH:
        1. Token embedding layer to convert tokens to vector.
        2. Positional embeding to add position information.
        3. Stack transformer blacks (The main computation).
        4. Final layer normalization and language modelling head.
        
        GPT ARCHITECTURE:
        tokens → embedding → + pos_embedding → transformer_blocks → layer_norm → lm_head → logits
        
        HINTS:
        - Positional embeddings are learned
        - Final layer norm stabilizes training
        - Language modelling head is a separate Linear(embed_dim, vocab_size) layer.
        (weight tying with the token embeddings not implemented yet).
        """
        self.config = config
        self.embedding_layer = EmbeddingLayer(config.vocab_size,
                                            config.embed_dim,
                                            config.max_seq_len,
                                            config.encoding_type)
        self.blocks = [TransformerBlock()
                    for _ in range(config.num_blocks)]
        pass