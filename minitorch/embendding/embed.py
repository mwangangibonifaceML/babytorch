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
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        
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
    
    
class PositionalEncoding:
    def __init__(self, max_seq_len: int, embed_dim: int) -> None:
        self.max_seq_len   : int = max_seq_len
        self.embed_dim     : int = embed_dim
    
        limit = math.sqrt(2.0 / self.embed_dim)
        self.pos_embedding = Tensor.rand(
            -limit, limit, (self.max_seq_len, self.embed_dim), requires_grad= True
        )
            
    def forward(self, X: Tensor) -> Tensor:
        """
        Positional Encoding forward

        Args:
            X (Tensor): Embedding Tensor to find positional encoding (Expects a 3D Tensor)

        Returns:
            Tensor: Position aware encoding
        """
        if len(X.shape) == 2:
            X = X.reshape(1, X.shape[0], X.shape[1])
            _, seq_len, embed_dim = X.shape
        #* verify if the seq_len and embed_dim of the input are within
        #* max_seq_len and embed_dim used in initializing the layer
        _, seq_len, embed_dim = X.shape
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length exceeds maximum: {seq_len} > {self.max_seq_len}\n"
                f"  ❌ Input sequence has {seq_len} positions, but max_seq_len is {self.max_seq_len}\n"
                f"  💡 Learned positional encodings have a fixed maximum length set at initialization\n"
                f"  🔧 Either truncate input to {self.max_seq_len} tokens, or create a new PositionalEncoding(max_seq_len={seq_len}, ...)"
            )
            
        if embed_dim != self.embed_dim:
            raise ValueError(
                f"Embedding dimension mismatch: input has {embed_dim}, expected {self.embed_dim}\n"
                f"  ❌ PositionalEncoding was created with embed_dim={self.embed_dim}, but input has embed_dim={embed_dim}\n"
                f"  💡 Token embeddings and positional encodings must have the same dimension to be added together\n"
                f"  🔧 Ensure your Embedding layer uses embed_dim={self.embed_dim}, or create PositionalEncoding(embed_dim={embed_dim}, ...)"
            )
        
        #* slice the embeddings using seq_len
        #* and batch batch dimension for addition to the word embeddings
        embedding = self.pos_embedding[:seq_len]
        pos_data = embedding.data[np.newaxis, :, :]
        pos_embedding_batched = Tensor(pos_data)
        
        #* add the positional embeddings to the word embeddings
        return pos_embedding_batched
        
    def __call__(self, X: Tensor) -> Tensor:
        "Allows call as a function"
        return self.forward(X)
    
    def parameters(self):
        "Get the parameters of the layer"
        return [self.pos_embedding]
    
    def __repr__(self) -> str:
        return f'PositionalEncoding(max_seq_len={self.max_seq_len}, embed_dim={self.embed_dim})'
    
    
class EmbeddingLayer:
    """
    Complete embedding system combining token and positional embeddings.

    Can handle the full embedding pipeline used in transformers and other sequence models.
    """
    def __init__(self,
                vocab_size: int,
                embed_dim: int,
                max_seq_len: int, 
                encoding_type: str | None = 'learned',
                scale_embeddings: bool = True) -> None:
        """
        Initialize complete embedding system.

        Create sub-components for token embedding and positional encoding

        **Insight**:
        Create token Embedding(vocab_size, embed_dim).
        
        Based on pos_encoding argument, create the appropriate positional encoder:
            - 'learned' -> PositionalEncoding(max_seq_len, embed_dim)
            - 'sinusoidal' -> create_sinusoidal_embeddings(max_seq_len, embed_dim)
            - None -> no positional encoding
        """
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.encoding_type = encoding_type
        self.scale_embeddings = scale_embeddings
        
        #* Token embedding layer
        self.token_embedding = Embedding(vocab_size, embed_dim)

        #* Positional encoding
        self.pos_encoding = PositionalEncoding(max_seq_len, embed_dim, encoding_type)
        
    def forward(self, tokens: Tensor):
        #* get the token embeddings and scale them if requested (transformer convention)
        token_embedding = self.token_embedding(tokens)
        if self.scale_embeddings:
            scale_factor = math.sqrt(self.embed_dim)
            token_embedding = token_embedding * scale_factor
            
        #* add positional embeddings to the token embeddings
        pos_embedding = self.pos_encoding(token_embedding)
        out = token_embedding + pos_embedding
        
        return out
    
    def __call__(self, tokens: Tensor)-> Tensor:
        "Allows call as a functions"
        return self.forward(tokens)
    
    def __repr__(self):
        return (f"EmbeddingLayer(vocab_size={self.vocab_size}, "
                f"embed_dim={self.embed_dim}, "
                f"pos_encoding_type='{self.encoding_type}')")
        
    def parameters(self) -> List[Tensor]:
        """Return all trainable parameters."""
        params = self.token_embedding.parameters()
        if self.encoding_type == 'learned':
            params.extend(self.pos_encoding.parameters())
        return params
    
    
def main():
    """🧪 Test complete embedding system."""
    print("🧪 Unit Test: Complete Embedding System...")

    #* Test 1: Learned positional encoding
    embed_learned = EmbeddingLayer(
        vocab_size=100,
        embed_dim=64,
        max_seq_len=128,
        encoding_type ='learned',
        scale_embeddings=False
    )

    tokens = Tensor([[1, 2, 3], [4, 5, 6]])
    output_learned = embed_learned.forward(tokens)

    assert output_learned.shape == (2, 3, 64), f"Expected shape (2, 3, 64), got {output_learned.shape}"

    #* Test 2: Sinusoidal positional encoding
    embed_sin = EmbeddingLayer(
        vocab_size=100,
        embed_dim=64,
        max_seq_len=128,
        encoding_type='sinusoidal',
        scale_embeddings=False
    )

    output_sin = embed_sin.forward(tokens)
    assert output_sin.shape == (2, 3, 64), "Sinusoidal embedding should have same shape"

    #* Test 3: No positional encoding
    embed_none = EmbeddingLayer(
        vocab_size=100,
        embed_dim=64,
        max_seq_len=128,
        encoding_type='learned',
        scale_embeddings=False
    )

    output_none = embed_none.forward(tokens)
    assert output_none.shape == (2, 3, 64), "No pos encoding should have same shape"

    #* Test 4: 1D input handling
    tokens_1d = Tensor([1, 2, 3])
    print(tokens_1d.shape)
    output_1d = embed_learned.forward(tokens_1d)

    assert output_1d.shape == (1,3, 64), f"Expected shape (1,3, 64) for 1D input, got {output_1d.shape}"

    #* Test 5: Embedding scaling
    embed_scaled = EmbeddingLayer(
        vocab_size=100,
        embed_dim=64,
        max_seq_len=128,
        scale_embeddings=True,
        encoding_type='learned',
    )

    #* Use same weights to ensure fair comparison
    embed_scaled.token_embedding.weight = embed_none.token_embedding.weight

    output_scaled = embed_scaled.forward(tokens)
    output_unscaled = embed_none.forward(tokens)

    #* Scaled version should be sqrt(64) times larger
    scale_factor = math.sqrt(64)
    expected_scaled = output_unscaled.data * scale_factor
    assert np.allclose(output_scaled.data, expected_scaled, rtol=1e-5), "Embedding scaling not working correctly"

    #* Test 6: Parameter counting
    params_learned = embed_learned.parameters()
    params_sin = embed_sin.parameters()
    params_none = embed_none.parameters()

    assert len(params_learned) == 2, "Learned encoding should have 2 parameter tensors"
    assert len(params_sin) == 1, "Sinusoidal encoding should have 1 parameter tensor"
    assert len(params_none) == 1, "No pos encoding should have 1 parameter tensor"

    print("✅ Complete embedding system works correctly!")
    
if __name__ == '__main__':
    main()