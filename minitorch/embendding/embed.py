import math
import numpy as np

from typing import List
from minitorch.tensor.tensor import Tensor
from minitorch.tokenization.tokenizer import BPETokenizer


class Embedding:
    def __init__(self, vocab_size: int, embed_dim: int):
        """
        Initialize Embedding weight with a Xavier-uniform initialization
        
        Create weight matrix of shape (vocab_size, embed_dim) using Xavier/Glorot
        initialization: sqrt(6.0 / (vocab_size + embed_dim))
        

        Args:
            vocab_size (int): The maximum number of allowed vocabulary 
            embed_dim (int): The number of embedding dimesions
        """
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        
        #* Xavier initialization for better gradient flow
        limit = math.sqrt(6.0 / (self.vocab_size + self.embed_dim))
        self.weight = Tensor.rand(
            -limit, limit,
            (self.vocab_size, self.embed_dim),
            requires_grad=True,
            )
        
    def forward(self, indices: Tensor) -> Tensor:
        """
        Forward pass: Look-up embeddings for a given indices

        Args:
            indices (Tensor): Token indices of shape (batch_size, seq_len) or (seq_len,)

        Returns:
            Tensor: Embedded vectors of shape (*indices.shape, embed_dim)
        """
        ids: np.ndarray = indices.data.astype(int)
        
        #* Handle invalid inputs
        min_idx: int = ids.min()
        max_idx: int = ids.max()
        
        if np.any(indices.data) >= self.vocab_size or np.any(indices.data) < 0:
                raise ValueError(
                f"Embedding index out of range for vocabulary size {self.vocab_size}\n"
                    f"  ❌ Found indices: min={min_idx}, max={max_idx} (valid range: 0 to {self.vocab_size - 1})\n"
                    f"  💡 Token IDs must be within the vocabulary. IDs >= vocab_size reference non-existent tokens\n"
                    f"  🔧 Check your tokenizer output, or increase vocab_size to at least {max_idx + 1}"
        
        )
        
        result = Tensor(
                    self.weight.data[indices.data.astype(int)],
                    requires_grad= self.weight.requires_grad,
                    _parents = (self.weight),
                    # device = self.weight.device,
                    )
        
        
        def _backward():
            if self.weight.requires_grad:
                grad_output = result.grad.reshape(-1, result.grad.shape[-1])
                indices_flattened = indices.data.astype(int).flatten()
                np.add.at(self.weight.grad, indices_flattened, grad_output)
                
        result._backward = _backward
        return result
    
    def __call__(self, indices: Tensor)-> Tensor:
        """
        Allow call as a python function

        Args:
            indices (Tensor): Token indices of shape (batch_size, seq_len) or (seq_len,)

        Returns:
            Tensor: Embedded vectors of shape (*indices.shape, embed_dim)
        """
        return self.forward(indices)
    
    def parameters(self) -> List[Tensor]:
        """Return trainable parameters."""
        return [self.weight]
    
    def __repr__(self):
        return f"Embedding(vocab_size={self.vocab_size}, embed_dim={self.embed_dim})"