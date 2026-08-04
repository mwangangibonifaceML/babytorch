from __future__ import annotations

import math
import time
import logging
import numpy as np
from typing import Tuple, Optional
from minitorch.tensor.tensor import Tensor
from minitorch.nn.layers import Linear
from minitorch.activations.activations import Softmax

#* Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _compute_attention_scores(query: Tensor, key: Tensor) -> Tensor:
    """Compute the attention scores for a single head.

    Args:
        query: A tensor of shape (batch_size, seq_length, head_dim) representing the query vectors.
        key: A tensor of shape (batch_size, seq_length, head_dim) representing the key vectors.
        value: A tensor of shape (batch_size, seq_length, head_dim) representing the value vectors.

    Returns:
        A tensor of shape (batch_size, seq_length, head_dim) representing the attention scores.
    """
    #* Compute the dot product of the query and key tensors O(n**2)
    scores = query @ key.transpose(-2,-1)  # (batch_size, seq_length, seq_length)
    # logger.info(f'Attention scores produced before scaling\n: {scores}')
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
    # logger.info(f'Attention scores produced after scaling but before masking\n: {scaled_scores}')
    return scaled_scores

def _apply_mask(scores: Tensor) -> Tensor:
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
    masked_scores = Tensor.masked_fill(scores)  # (batch,seq_length, seq_length)
    # logger.info(f'Attention scores produced after scaling and masking\n: {masked_scores}')
    return masked_scores

def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor, 
    value: Tensor,
    ) -> Tuple[Tensor, Tensor]:
    """Compute the scaled dot-product attention.

    Args:
        query: A tensor of shape (batch_size, seq_length, head_dim) representing the query vectors.
        key: A tensor of shape (batch_size, seq_length, head_dim) representing the key vectors.
        value: A tensor of shape (batch_size, seq_length, head_dim) representing the value vectors.

    Returns:
        A tuple of output, attention_weights where: output is of 
        shape (batch_size, seq_length, head_dim) representing the
        output of the attention mechanism, and attention_weights 
        is of shape (batch_size, seq_length, seq_length) representing the attention weights.
    """
    softmax_fn = Softmax()
    scores = _compute_attention_scores(query, key)  # (batch_size, seq_length, seq_length)
    
    #* scale for stability
    scaled_scores = _scale_scores(scores, query.shape[-1])  # (batch_size, seq_length, seq_length)
    masked_scores = _apply_mask(scaled_scores)  # (batch_size, seq_length, seq_length)
    
    #* convert to probabilities
    attention_weights = softmax_fn(masked_scores)  # (batch_size, seq_length, seq_length)
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
        attended, atten_wei = scaled_dot_product_attention(Q,K,V)

        #* merge heads back together
        concatenated_output = self._merge_heads(attended)
        
        #* pass through the output projection
        output = self.out_proj(concatenated_output)
        return output, atten_wei
    
    def __call__(self, X: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        return self.forward(X, mask)
    
    def parameters(self) -> list[Tensor]:
        """
        Return all trainable parameters by collecting parameters from all linear layers,
        
        Expects 4 parameters when projections are instatiated using bias=False (each projection
        with weight only, 4 * 1 = 4), else expects 8 parameters, (each projection with weight and bias, 4 * 2 = 8)
        
        APPROACH:
        1. Get parameters from q_proj, k_proj, v_proj, out_proj
        2. Combine into single list

        Returns:
            List of all parameter tensors

        """
        params = []
        params.extend([self.q_proj.parameters()] )
        params.extend([self.k_proj.parameters()] )
        params.extend([self.v_proj.parameters()] )
        params.extend([self.out_proj.parameters()] )
        
        return  params
    
    
def test_unit_scaled_dot_product_attention():
    """🧪 Test scaled dot-product attention implementation."""
    logger.info("🧪 Unit Test: Scaled Dot-Product Attention...\n")
    
    #* Test basic functionality
    batch, seq_len, d_model = 2,4,8
    Q = Tensor(np.random.randn(batch, seq_len, d_model))
    K = Tensor(np.random.randn(batch, seq_len, d_model))
    V = Tensor(np.random.randn(batch, seq_len, d_model))
    
    output, att_wei = scaled_dot_product_attention(Q,K,V)
    
    #* check the output shapes
    assert output.shape == (batch,seq_len,d_model), f'Output shape {output.shape} incorrect'
    assert att_wei.shape == (batch,seq_len,seq_len), f'Output shape {att_wei.shape} incorrect'
    
    #* check if the attention weights sum to 1
    weight_sum = att_wei.sum(axis=-1, keepdims=True)
    
    #* the shape of attenion weights remains the same other than the last dim(turns to 1)
    expected_sum = np.ones((att_wei.shape[0], att_wei.shape[1],1))
    assert np.allclose(weight_sum.data, expected_sum.data, atol=1e-6), 'Attention weights do not sum to 1.'
    
    #* check whether future positions are masked(have zero)
    for b in range(batch):
        for i in range(seq_len):
            for j in range(i+1, seq_len):
                assert abs(att_wei[b,i,j].data < 1e-6), f'Future attention not masked at ({i}, {j})'
                
    logger.info("✅ Scaled dot product works perfectly\n")
    
def test_unit_split_heads():
    """🧪 Test head splitting reshape."""
    logger.info("🧪 Unit Test: Split Heads...")
    batch, seq_len, embed_dim, n_heads = (2,10,64,8)
    mha = MultiHeadAttention(embed_dim, n_heads)
    X = Tensor.randn((batch, seq_len, embed_dim), requires_grad=True)
    splits = mha._split_heads(X) #* (batch, seq_len, embed_dim) -> (batch, n_heads, se_len, head_dim)
    assert splits.shape == (batch, n_heads, seq_len, embed_dim/n_heads), f'Expected {batch, n_heads, seq_len, embed_dim/n_heads} got {split.shape}'
    logger.info("✅ Split heads works perfectly: Correct 4D shape!\n")
    
def test_unit_merge_heads():
    """🧪 Test head merging..."""
    logger.info("🧪 Unit Test: Merge Heads...")
    batch, seq_len, embed_dim, n_heads = (2,10,64,8)
    mha = MultiHeadAttention(embed_dim, n_heads)
    X = Tensor.randn((batch, seq_len, embed_dim), requires_grad=True)
    splits = mha._split_heads(X) #* (batch, seq_len, embed_dim) -> (batch, n_heads, se_len, head_dim)
    merges = mha._merge_heads(splits) #*(batch, n_heads, seq_len, head_dims) -> (batch, seq_len, embed_dim)
    assert merges.shape == (batch, seq_len, embed_dim), f'Expected {batch,seq_len, embed_dim} got {merges.shape}'
    logger.info("✅ Merge heads works perfectly: Correct 3D shape!\n")
    
def test_unit_mutliheadattention():
    """🧪 Test multi-head attention implementation."""
    logger.info("🧪 Unit Test: Multi-Head Attention...")
    batch, seq_len, embed_dim, n_heads = (2,10,64,8)
    mha = MultiHeadAttention(embed_dim, n_heads)
    
    assert mha.embed_dim == embed_dim
    assert mha.n_heads == n_heads
    assert mha.head_dim == embed_dim // n_heads
    
    #* test the len of the parameters
    params = mha.parameters()
    assert len(params) == 4, f'Expected 4 parameters, got {len(params)}'
    
    #* test forward pass
    x = Tensor.randn((batch, seq_len, embed_dim))
    output,_ = mha(x)
    assert output.shape == (batch, seq_len, embed_dim), f'Expected {batch, seq_len, embed_dim}, got {output.shape}'
    
    #* test different head configurations
    mha_small = MultiHeadAttention(embed_dim=32, n_heads=4)
    x_small = Tensor.randn((1,5,32))
    output_small,_ = mha_small(x_small)
    assert output_small.shape == (1,5,32), f'Expected {1,5,32}, got {output_small.shape}'
    logger.info('✅ Multihead attention works correctly.\n')
        
def analyze_attention_complexity():
    """📊 Analyze attention computational complexity and memory scaling."""
    logger.info("📊 Analyzing Attention Complexity...\n")
    
    logger.info("\nSequence Length vs Attention Matrix Size:")
    logger.info("Seq Len | Attention Matrix | Memory (KB) | Complexity")
    logger.info("-" * 55)
    embed_dim = 64
    sequence_lengths = [16, 32, 64, 128, 256, 512, 1024]
    
    for seq_len in sequence_lengths:
        #* attention matrix size
        attention_matrix_size = seq_len * seq_len
        
        #* memory for the attention weight
        attention_memory_kb = (attention_matrix_size * 4) /1024
        
        #* total complexity (Q@k + softmax + weights@V)
        complexity = 2 * seq_len * seq_len * embed_dim + seq_len * seq_len
        logger.info(f"{seq_len:7d} | {attention_matrix_size:14d} | {attention_memory_kb:10.2f} | {complexity:10.0f}")

    logger.info(f"\n💡 KEY INSIGHT: Attention memory scales as O(n^2) with sequence length")
    logger.info(f"🚀 For seq_len=1024, attention matrix alone needs {(1024*1024*4)/1024/1024:.1f} MB\n")
    
def analyze_attention_timing():
    """📊 Measure attention computation time vs sequence length."""
    logger.info("\n📊 Analyzing Attention Timing...")

    embed_dim, num_heads = 64, 8
    sequence_lengths = [32, 64, 128, 256, 512, 1024]

    logger.info("\nSequence Length vs Computation Time:")
    logger.info("Seq Len | Time (ms) | Ops/sec | Scaling")
    logger.info("-" * 40)

    prev_time = None
    for seq_len in sequence_lengths:
        # Create test input
        x = Tensor.randn((1, seq_len, embed_dim))
        mha = MultiHeadAttention(embed_dim, num_heads)

        # Time multiple runs for stability
        times = []
        for _ in range(5):
            start_time = time.time()
            _ = mha.forward(x)
            end_time = time.time()
            times.append((end_time - start_time) * 1000)  # Convert to ms

        avg_time = np.mean(times)
        ops_per_sec = 1000 / avg_time if avg_time > 0 else 0

        # Calculate scaling factor vs previous
        scaling = avg_time / prev_time if prev_time else 1.0

        logger.info(f"{seq_len:7d} | {avg_time:8.2f} | {ops_per_sec:7.0f} | {scaling:6.2f}x")
        prev_time = avg_time

    logger.info(f"\n💡 KEY INSIGHT: Attention time scales roughly as O(n^2) with sequence length")
    logger.info(f"🚀 This is why attention efficiency techniques are an active area of research\n")
    
def analyze_attention_memory_overhead():
    """📊 Analyze memory overhead during training (forward + backward passes)."""
    logger.info("\n📊 Analyzing Attention Memory Overhead During Training...")

    embed_dim, num_heads = 128, 8
    sequence_lengths = [128, 256, 512, 1024]

    logger.info("\nMemory Overhead Analysis (Training vs Inference):")
    logger.info("Seq Len | Forward | + Gradients | + Optimizer | Total Memory")
    logger.info("-" * 65)

    for seq_len in sequence_lengths:
        # Forward pass memory (attention matrix)
        attention_matrix_mb = (seq_len * seq_len * 4) / (1024 * 1024)

        # Backward pass adds gradient storage (1× forward: one gradient tensor)
        backward_memory_mb = attention_matrix_mb

        # Optimizer state (Adam: +2× for momentum and velocity, incremental)
        optimizer_memory_mb = 2 * attention_matrix_mb

        # Total = forward + gradients + optimizer state
        total_memory_mb = attention_matrix_mb + backward_memory_mb + optimizer_memory_mb

        logger.info(f"{seq_len:7d} | {attention_matrix_mb:6.2f}MB | {backward_memory_mb:10.2f}MB | {optimizer_memory_mb:10.2f}MB | {total_memory_mb:11.2f}MB")

    logger.info(f"\n💡 KEY INSIGHT: Training requires ~4x memory of inference (1x forward + 1x gradients + 2x optimizer state)")
    logger.info(f"🚀 For GPT-3 (96 layers, 2048 context): ~6GB just for attention gradients!\n")


if __name__ == '__main__':
    test_unit_scaled_dot_product_attention()
    test_unit_split_heads()
    test_unit_merge_heads()
    test_unit_mutliheadattention()
    analyze_attention_complexity()
    analyze_attention_timing()
    analyze_attention_memory_overhead()