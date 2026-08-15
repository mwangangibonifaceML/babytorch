from __future__ import annotations

import numpy as np

from typing import Dict, Any
from minitorch.tensor.tensor import Tensor
from minitorch.activations.activations import GELU, ReLU, Softmax
from minitorch.nn.layers import Linear, LayerNormalization, Module, Dropout, Parameter, Sequential
from minitorch.attention.attention import MultiHeadAttention
from minitorch.embendding.embed import Embedding, PositionalEncoding
from minitorch.losses.losses import SoftMaxCrossEntropy

#ghhjkkjgf
class FeedForward(Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.layer1 = Linear(embed_dim, 4 * embed_dim)
        self.activ = GELU()
        self.layer2 = Linear(4 * embed_dim, embed_dim)
        
    def __call__(self, inputs: Tensor) -> Tensor:
        "Pass the inputs through the network"
        return self.layer2(self.activ(self.layer1(inputs)))

    def __repr__(self) -> str:
        return repr(self.net)
    
    def parameters(self) -> list[Parameter]:
        params = []
        params.extend(self.layer1.parameters())
        params.extend(self.layer2.parameters())
        return params

class Block(Module):
    def __init__(self, embed_dim: int, n_heads: int, dim: int, dropout: float):
        super().__init__()
        self.mha = MultiHeadAttention(embed_dim, n_heads)
        self.ln1 = LayerNormalization(dim)
        self.dropout = Dropout(dropout)
        self.ffd = FeedForward(embed_dim)
        self.ln2 = LayerNormalization(dim)
        
        
    def __call__(self, input: Tensor) -> Tensor:
        #* Attention sub-layer
        X = self.mha(self.ln1(input))
        X = input + self.dropout(X)
        
        #* Feedforward sub-layer
        X = self.ffd(self.ln2(X))
        return X + self.dropout(X)
    
    def parameters(self) -> list[Parameter]:
        params = []
        params.extend(self.mha.parameters())
        params.extend(self.ln1.parameters())
        params.extend(self.ln2.parameters())
        params.extend(self.ffd.parameters())
        return params

    def __repr__(self) -> str:
        return f'''
Block(Multi Head Attention= {self.mha}),
    Layer Normalization = {repr(self.ln)},
    Feed Forwad = {repr(self.ffd)}\n'''
        
class GPT(Module):
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
        super().__init__()
        self.config = config
        
        #* create the embedding table and positional encodings
        self.wte = Embedding(config['vocab_size'], config['embed_dim'])
        self.wpe = PositionalEncoding(config['max_seq_length'],config['embed_dim'])
        
        #* create the blocks\
        self.blocks = Sequential(
            *[
            Block(
                config['embed_dim'],
                config['n_heads'],
                config['dim'], 
                config['dropout'])
                for _ in range(config['n_layers'])
            ]
        )
                                
        #* post block normalization
        self.ln_f = LayerNormalization(config['embed_dim'])
        
        #* create the output projection(LM Head)
        self.lm_head = Linear(config['embed_dim'], config['vocab_size'])
        
        #* weight tying
        self.lm_head.weight = self.wte.weight.transpose()
        
    def forward(self,idx: Tensor, targets: Tensor|None = None) -> tuple[Tensor, Tensor|None]:
        """
        Pass the inputs through the GPT Model and calculat the model's loss
        during the training phase
        
        Args:
            idx (Tensor): token ids
            targets (Optional|Tensor): Ground truth | Expected predictions
            
        Returns:
            Tuple[Tensor,Tensor]: The calculated logits and the loss if training, else just logits
        """
        wte = self.wte(idx)
        wpe = self.wpe(wte)
        token_embed = wte + wpe
        x = self.blocks(token_embed)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        
        if targets is not None:
            loss = SoftMaxCrossEntropy()(logits, targets)
            return logits, loss
        return logits, loss
    
    def __call__(self,idx: Tensor, targets: Tensor|None = None) -> tuple[Tensor, Tensor|None]:
        return self.forward(idx, targets)
    
    def generate(self, idx:Tensor, max_new_tokens: int) -> Tensor:
        "Generate using the trained model"
        for _ in range(max_new_tokens):
            logits,_ = self.forward(idx)
            logits = logits[:,-1,:]
            probs = Softmax()(logits)
            next_token = np.array([
                        np.random.choice(probs.data.shape[1], p=row)
                        for row in probs.data
                        ])[:, None]
            idx = Tensor(np.concatenate([idx.data, next_token], axis=1))
        return idx
    
    def parameters(self) -> list[Parameter]:
        params = []
        params.extend(self.wte.parameters())
        params.extend(self.wpe.parameters())
        params.extend(self.blocks.parameters())  
        params.extend(self.ln_f.parameters())
        return params

    def __repr__(self) -> str:
        return f'''GPT Model:
    - Embedding Dimension: {self.config['embed_dim']}
    - Number of Heads: {self.config['n_heads']}
    - Number of Layers: {self.config['n_layers']}
    - Context Length: {self.config['max_seq_length']}
    '''