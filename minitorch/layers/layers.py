from __future__ import annotations
import numpy as np
from typing import Any
from minitorch.activations.activations import ReLU, Sigmoid, Tanh
from minitorch.tensor.tensor import Tensor
from typing import Union, Tuple, List

#* constants for weight initialization
XAVIER_SCALE_FACTOR = 1.0 # Xavier/Glorot init uses sqrt(1/in_features)
HE_SCALE_FACTOR = 2.0 # He init uses sqrt(2/in_features) for ReLU

#* constants for dropout
DROPOUT_MIN_PROB = 0.0 # drop nothing
DROPOUT_MAX_PROB = 1.0 # drop everything

TOLERANCE = 1e-3 # numerical tolerance for floating point comparisons

class Layer:
    """Base class for all layers in the neural network.
    
    All layers should inherit from this class and implement the forward and backward methods.
    
    Methods:
        forward(input): Computes the forward pass of the layer.
        backward(grad_output): Computes the backward pass of the layer.
        parameters(): Returns the parameters of the layer.
        The __call__ method is provided to allow instances of Layer to be called like functions.
    """
    
    def forward(self, input: Tensor) -> Tensor:
        """Forward pass throught the layer.
        
        Args:
            input (Tensor): The input tensor.
        
        Returns:
            Tensor: The output tensor.
        """
        raise NotImplementedError("Forward method not implemented.")
    
    def __call__(self, input: Tensor,  *args: Any, **kwds: Any) -> Tensor:
        """Allows the layer to be called like a function.
        
        Args:
            input (Tensor): The input tensor.
            *args: Additional positional arguments.
            **kwds: Additional keyword arguments.
        
        Returns:
            Tensor: The output tensor from the forward pass.
        """
        return self.forward(input, *args, **kwds)
    
    def parameters(self) -> list[Tensor]:
        """Returns the parameters of the layer.
        
        Returns:
            list[Tensor]: A list of tensors representing the parameters of the layer.
        """
        return [] #* Base layer has no parameters
    
    def __repr__(self) -> str:
        """String representation of the layer."""
        return f"{self.__class__.__name__}()"
    
class Linear(Layer):
    """A fully connected linear layer: implements: y = xW + b. 
    
    Applies a linear transformation to the incoming data.
    
    Approach:
        - create weight matrix of shape (out_features, in_features)
        - create bias vector of shape (out_features), initialized to zeros if bias=True
        - set requires_grad=True for both weight and bias tensors
        
    HINTS:
        - Xavier init: scale = sqrt(1/in_features)
        - use np.random.randn to create weight and bias tensors
        -
    
    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
    
    Methods:
        forward(input): Computes the forward pass of the linear layer.
        parameters(): Returns the parameters of the linear layer.
    """
    
    def __init__(self, in_features: int, out_features: int, bias:bool=False) -> None:
        self.in_features = in_features
        self.out_features = out_features
        
        #* Xavier initialization for stable gradients
        scale = (XAVIER_SCALE_FACTOR / in_features) ** 0.5
        weight_data = np.random.randn(out_features, in_features) * scale
        self.weight = Tensor(weight_data, requires_grad=True)
        
        if bias == True:
            bias_data = np.zeros(out_features)
            self.bias = Tensor(bias_data, requires_grad=True)
            
        else:
            self.bias = None
    
    def forward(self, input: Tensor) -> Tensor:
        """Forward pass through the linear layer.
        
        Args:
            input (Tensor): The input tensor of shape (batch_size, in_features).
        
        Returns:
            Tensor: The output tensor of shape (batch_size, out_features).
        """
        if self.bias is None:
            return input.matmul(self.weight.transpose())
        else:
            return input.matmul(self.weight.transpose()) + self.bias
    
    def parameters(self) -> list[Tensor]:
        """Returns the parameters of the linear layer.
        
        Returns:
            list[Tensor]: A list containing the weight and bias tensors.
        """
        parameters = [self.weight]
        if self.bias is not None:
            parameters.append(self.bias)
        return parameters

    def __repr__(self) -> str:
        """String representation of the linear layer."""
        bias_str = ", bias=True" if self.bias is not None else ""
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}{bias_str})"
    
class Dropout(Layer):
    """Initialize dropout layer.
    Args:
        p (float): Probability of dropping a unit. Must be in the range [0.0, 1.0).
    """
    def __init__(self, p: float) -> None:
        if not (DROPOUT_MIN_PROB <= p < DROPOUT_MAX_PROB):
            raise ValueError(f"Dropout probability must be in the range [{DROPOUT_MIN_PROB}, {DROPOUT_MAX_PROB}), got {p}.")
        self.p = p
        
    def forward(self, input: Tensor, training: bool=False) -> Tensor:
        """Forward pass through the dropout layer.
        
        During training, randomly sets a fraction p of inputs to zero.
        During inference, passes the input unchanged.
        
        Approach:
            - If training mode=False or p=0, return input unchanged
            - If p=1, return a tensor of zeros with the same shape as input
            - If training mode = True and p > 0, set random elements to 0 with
            probability p, and scale remaining elements by 1/(1-p)
            - If in evaluation mode, return input unchanged
        Hints:
            - Use np.random.random() < keep_prob to create dropout mask
            - training mode=False should return input unchanged
        """
            
        #* during evaluation or no dropout, return input unchanged
        if not training or self.p == DROPOUT_MIN_PROB:
            return input
        
        if self.p == DROPOUT_MAX_PROB:
            return Tensor(np.zeros_like(input.data), requires_grad=input.requires_grad)
        
        #* during training, apply dropout
        keep_prob = DROPOUT_MAX_PROB - self.p
        mask = (np.random.random(input.data.shape) < keep_prob)
        
        #* apply mask and scale to preserve gradients
        mask_tensor = Tensor(mask.astype(np.float32), requires_grad=False)
        scale = Tensor(np.array(1.0 / keep_prob), requires_grad=False)
        output = input * mask_tensor * scale
        output.requires_grad = input.requires_grad
        return output
    
    def parameters(self) -> list[Tensor]:
        return super().parameters()
    
    def __repr__(self) -> str:
        return f"Dropout(p={self.p})"
    
class Sequential:
    """" A container that chains layers together sequentially. It chains these layers
    together, calling forward() on each layer in sequence. This is similar to 
    torch.nn.Sequential but much simpler.
    
    Args:
        layers (list[Layer]): A list of Layer instances to be chained together.
    
    """
    def __init__(self, *layers: Tuple[list[Layer]]):
        """Initialize the Sequential container with a list of layers.
        Accepts both (layer1, layer2, ...) and ([layer1, layer2, ...]) formats.
        """
        if len(layers) == 1 and isinstance(layers[0], (list, tuple)):
            self.layers = list(layers[0])
        else:
            self.layers = list(layers)
            
    def forward(self, input: Tensor) -> Tensor:
        """Forward pass through all layers in sequence."""
        for layer in self.layers:
            input = layer(input)
        return input
        
    def __call__(self, input: Tensor) -> Tensor:
        """Allows the Sequential container to be called like a function."""
        return self.forward(input)
    
    def parameters(self) -> list[Tensor]:
        """Returns the parameters of all layers in the Sequential container."""
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params
        
    def __repr__(self) -> str:
        layers_repr = ",\n  ".join(repr(layer) for layer in self.layers)
        return f"Sequential(\n  {layers_repr}\n)"
    
    

    

    

def test_gradient_preparation_linear():
    """🔬 Test Linear layer is ready for gradients (Module 05)."""
    print("🔬 Gradient Preparation Test: Linear Layer...")

    layer = Linear(10, 5, bias=True)

    # Verify requires_grad is set
    assert layer.weight.requires_grad == True, "Weight should require gradients"
    assert layer.bias.requires_grad == True, "Bias should require gradients"
    print("🧪 requires_grad tests passed ...")
    
    # Verify gradient placeholders exist (even if None initially)
    assert hasattr(layer.weight, 'grad'), "Weight should have grad attribute"
    assert hasattr(layer.bias, 'grad'), "Bias should have grad attribute"
    print("🧪 Gradient attributes tests passed ...")
    
    # Verify parameter collection works
    params = layer.parameters()
    assert len(params) == 2, "Should return 2 parameters"
    assert all(p.requires_grad for p in params), "All parameters should require gradients"
    print("🧪 Gradient preparation tests passed ...")
    
    print("✅ Layer ready for gradient-based training!")
    print("="*50)

if __name__ == "__main__":
    print("Running unit tests for the layers...")
    unit_test()
    test_edge_cases_linear()
    test_gradient_preparation_linear()
    test_unit_dropout()
    print("All tests passed!")
    print("="*50)