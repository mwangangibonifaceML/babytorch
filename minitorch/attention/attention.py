from __future__ import annotations

import math
import numpy as np
from typing import Tuple, Optional
from minitorch.tensor.tensor import Tensor
from minitorch.nn.layers import Linear

def _compute_attention_scores(query: Tensor, key: Tensor) -> Tensor:
    """Compute the attention scores for a single head.

    Args:
        query: A tensor of shape (batch_size, seq_length, head_dim) representing the query vectors.
        key: A tensor of shape (batch_size, seq_length, head_dim) representing the key vectors.
        value: A tensor of shape (batch_size, seq_length, head_dim) representing the value vectors.

    Returns:
        A tensor of shape (batch_size, seq_length, head_dim) representing the attention scores.
    """
    #* Compute the dot product of the query and key tensors
    scores = query @ key.transpose(2, 3)  # (batch_size,num_heads, seq_length, seq_length)
    return scores

def _scale_scores(scores: Tensor, head_dim: int) -> Tensor:
    """Scale the attention scores by the square root of the head dimension.

    Args:
        scores: A tensor of shape (batch_size, seq_length, seq_length) representing the attention scores.
        head_dim: An integer representing the dimension of each head.

    Returns:
        A tensor of shape (batch_size, seq_length, seq_length) representing the scaled attention scores.
    """
    #* Scale the scores by the square root of the head dimension
    scaled_scores = scores / math.sqrt(head_dim)
    return scaled_scores

def _apply_mask(scores: Tensor, mask: Tensor) -> Tensor:
    """Apply a mask to the attention scores to prevent attending to future tokens.

    Args:
        scores: A tensor of shape (batch_size, seq_length, seq_length) representing the attention scores.
        mask: A tensor of shape (seq_length, seq_length) representing the mask.

    Returns:
        A tensor of shape (batch_size, seq_length, seq_length) representing the masked attention scores.


    Returns:
        A tensor of shape (batch_size, seq_length, seq_length) representing the masked attention scores.
    """
    #* Create a mask to prevent attending to future tokens
    mask = Tensor.tril(mask, diagonal=0)  # (seq_length, seq_length)
    masked_scores = scores + (Tensor.ones(mask.shape) - mask) * -1e9  # (batch_size, seq_length, seq_length)
    return masked_scores
    
def _softmax(scores: Tensor) -> Tensor:
    """Apply the softmax function to the attention scores.

    Args:
        scores: A tensor of shape (batch_size, seq_length, seq_length) representing the attention scores.

    Returns:
        A tensor of shape (batch_size, seq_length, seq_length) representing the attention weights.
    """
    #* Apply the softmax function to the scores
    exp_scores = np.exp(scores.data)  # (batch_size, seq_length, seq_length)
    attention_weights = exp_scores / exp_scores.sum(axis=-1, keepdims=True)  # (batch_size, seq_length, seq_length)
    return Tensor(attention_weights)

def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor, 
    value: Tensor,
    mask: Optional[Tensor]= None) -> Tuple[Tensor, Tensor]:
    """Compute the scaled dot-product attention.

    Args:
        query: A tensor of shape (batch_size, seq_length, head_dim) representing the query vectors.
        key: A tensor of shape (batch_size, seq_length, head_dim) representing the key vectors.
        value: A tensor of shape (batch_size, seq_length, head_dim) representing the value vectors.
        mask: A tensor of shape (seq_length, seq_length) representing the mask.

    Returns:
        A tuple of output, attention_weights where: output is of 
        shape (batch_size, seq_length, head_dim) representing the
        output of the attention mechanism, and attention_weights 
        is of shape (batch_size, seq_length, seq_length) representing the attention weights.
    """
    scores = _compute_attention_scores(query, key)  # (batch_size, seq_length, seq_length)
    scaled_scores = _scale_scores(scores, query.shape[-1])  # (batch_size, seq_length, seq_length)
    if mask is not None:
        masked_scores = _apply_mask(scaled_scores, mask)  # (batch_size, seq_length, seq_length)
        attention_weights = _softmax(masked_scores)  # (batch_size, seq_length, seq_length)
        output = attention_weights @ value  # (batch_size, seq_length, head_dim)
        
    attention_weights = _softmax(scaled_scores)  # (batch_size, seq_length, seq_length)
    output = attention_weights @ value  # (batch_size, seq_length, head_dim)
    
    return output, attention_weights

class MultiHeadAttention:
    """
    MultiHead Attention mechanism, which run multiple heads
    in parallel, each head learning a different relationship
    from the other heads.
    """
    def __init__(self, embed_dim: int, n_heads: int)-> None:
        """
        Initialize multi-head attention.

        Set up linear projections and validate configuration

        Args:
            embed_dim (int): Embedding dimension of the input and output tensors
            n_heads (int): Number of parallel heads to run in the attention mechanism
        """
        assert embed_dim % n_heads == 0, "Embedding dimension must be divisible by number of heads"
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads

        #* Initialize linear projections for query, key, and value
        self.q_proj = Linear(embed_dim, embed_dim)
        self.k_proj = Linear(embed_dim, embed_dim)
        self.v_proj = Linear(embed_dim, embed_dim)
        
        #* Output linear projection
        self.out_proj = Linear(embed_dim, embed_dim)        
        
    def _split_heads(self, X: Tensor) -> Tensor:
        """
        Split the input tensor into multiple heads.

        Args:
            X (Tensor): Input tensor of shape (batch_size, seq_length, embed_dim)
            seq_len (int): Sequence length of the input tensor

        Returns:
            Tensor: Reshaped tensor of shape (batch_size, n_heads, seq_length, head_dim)
        """
        batch_size , seq_len = X.shape[0], X.shape[1]
        #* Reshape and permute to get (batch_size, n_heads, seq_length, head_dim)
        X = X.reshape(batch_size, seq_len, self.n_heads, self.head_dim)  # (batch_size, seq_length, n_heads, head_dim)
        return X.transpose(dim0=1, dim1=2)  # (batch_size, n_heads, seq_length, head_dim)
    
    def _merge_heads(self, X: Tensor) -> Tensor:
        """
        Merge the multiple heads back into a single tensor.

        Args:
            X (Tensor): Input tensor of shape (batch_size, n_heads, seq_length, head_dim)

        Returns:
            Tensor: Reshaped tensor of shape (batch_size, seq_length, embed_dim)
        """
        batch_size , seq_len = X.shape[0], X.shape[2]
        #* Permute and reshape to get (batch_size, seq_length, embed_dim)
        X = X.transpose(dim0=1, dim1=2)  # (batch_size, seq_length, n_heads, head_dim)
        return X.reshape(batch_size, seq_len, self.embed_dim)  # (batch_size, seq_length, embed_dim)
    
    def forward(self, X: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Perform the forward pass of multi-head attention.

        Args:
            query (Tensor): Input tensor for queries of shape (batch_size, seq_length, embed_dim)
            key (Tensor): Input tensor for keys of shape (batch_size, seq_length, embed_dim)
            value (Tensor): Input tensor for values of shape (batch_size, seq_length, embed_dim)
            mask (Tensor): Mask tensor of shape (seq_length, seq_length) to prevent attending to certain positions

        Returns:
            Tensor: Output tensor of shape (batch_size, seq_length, embed_dim) after applying multi-head attention
        """
        #* Linear projections
        Q = self.q_proj(X)    # (batch_size, seq_length, embed_dim)
        K = self.k_proj(X)    # (batch_size, seq_length, embed_dim)
        V = self.v_proj(X)    # (batch_size, seq_length, embed_dim)
        
        #* split the heads
        Q = self._split_heads(Q) #* (Q.shape: (batch_size, n_heads, seq_length, head_dim))
        K = self._split_heads(K) #* (K.shape: (batch_size, n_heads, seq_length, head_dim))
        V = self._split_heads(V) #* (V.shape: (batch_size, n_heads, seq_length, head_dim))
        
        #* apply the scaled-dot-product attention
        if mask and len(mask.shape) == 3:
            mask_batch_size,mask_seq_len,_ = mask.shape
            mask_reshaped = mask.reshape(mask_batch_size, 1, mask_seq_len,mask_seq_len)
            attended, _ = scaled_dot_product_attention(Q,K,V, mask_reshaped)
        else:
            attended, _ = scaled_dot_product_attention(Q,K,V)
        #  scores = query @ key.transpose(1,2)  
        #* merge heads back together
        concatenated_output = self._merge_heads(attended)
        
        #* pass through the output projection
        output = self.out_proj(concatenated_output)
        return output
    
    def __call__(self, X: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        return self.forward(X, mask)
    
    def parameters(self) -> list[Tensor]:
        """
        Return all trainable parameters by collecting parameters from all linear layers

        APPROACH:
        1. Get parameters from q_proj, k_proj, v_proj, out_proj
        2. Combine into single list

        Returns:
            List of all parameter tensors

        """
        params = []
        params.extend([self.q_proj.weight] )
        params.extend([self.q_proj.weight] )
        params.extend([self.q_proj.weight] )
        params.extend([self.q_proj.weight] )
        
        return  params