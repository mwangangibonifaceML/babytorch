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
EPSILON = 1e-5 # layer norm epsilon to shift the standard deviation

def _get_modules(obj) -> list['Module']:
    """
    A simple recursive function that finds all Module objects within any given
    object by searching through its attributes, lists, tuples, and dicts.
    """
    modules = []
    if isinstance(obj, Layer):
        return [obj]
        
    if isinstance(obj, dict):
        for value in obj.values():
            modules.extend(_get_modules(value))

    if isinstance(obj, (list, tuple)):
        for item in obj:
            modules.extend(_get_modules(item))

    return modules

class Parameter(Tensor):
    """A trainable Tensor.\n
    
    Represents trainable parameters such as weights and biases.\n
    Always participates in gradient computations

    Args:
        Tensor (Tensor): Input tensor to convert to a parameter
    """
    def __init__(self, data: Tensor):
        assert isinstance(data, Tensor), 'Data must be an instance of Tensor.'
        super().__init__(data.data, requires_grad=True)
        
    def zero_grad(self):
        if self.grad is not None:
            self.grad = np.zeros_like(self.data)
            
    def detach(self):
        raise RuntimeError(
            'Cannot detach a Parameter from the computational graph.',
            'Convert to Tensor if intentional'
        )
        
    def __repr__(self) -> str:
        return (
            f'Parameter(Tensor(data={self.data}), \n'
            f'shape={self.shape}, \n'
            f'requires_grad={self.requires_grad}'
                )


class Layer:
    """Base class for all layers in the neural network.
    
    All layers should inherit from this class and implement the forward and backward methods.
    
    Methods:
        forward(input): Computes the forward pass of the layer.
        backward(grad_output): Computes the backward pass of the layer.
        parameters(): Returns the parameters of the layer.
        The __call__ method is provided to allow instances of Layer to be called like functions.
    """
    def __init__(self) -> None:
        self.training =True
        
    def eval(self):
        self.training = False
        for module in _get_modules(self):
            module.training = False
        
    def train(self):
        self.training = True
        for module in _get_modules(self):
            module.training = True
        
    
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
    
    def parameters(self) -> list[Parameter]:
        """Returns the parameters of the layer.
        
        Returns:
            list[Tensor]: A list of tensors representing the parameters of the layer.
        """
        params = []

        for value in self.__dict__.values():

            if isinstance(value, Parameter):
                params.append(value)

            elif isinstance(value, Layer):
                params.extend(value.parameters())

            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, Parameter):
                        params.append(item)
                    elif isinstance(item, Layer):
                        params.extend(item.parameters())

        return params
    
    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()
    
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
        
    Attributes:
        weight (Parameter): The learnable weights of the module of shape
                            `(in_features, out_features)`.
        bias (Parameter):   The learnable bias of the module
                            of shape `(out_features,)`.
    
    Methods:
        forward(input): Computes the forward pass of the linear layer.
        parameters(): Returns the parameters of the linear layer.
    """
    
    def __init__(self, in_features: int, out_features: int, bias=False) -> None:
        self.in_features = in_features
        self.out_features = out_features
        
        #* Xavier initialization for stable gradients
        scale = (XAVIER_SCALE_FACTOR / in_features) ** 0.5
        self.weight = Parameter(
            Tensor(np.random.randn(out_features, in_features) * scale))
        
        if bias == True:
            self.bias = Parameter(Tensor(np.zeros(out_features)))
            
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
            product = input.matmul(self.weight.transpose())
            return product + self.bias
    
    def parameters(self) -> list[Tensor]:
        """Returns the parameters of the linear layer.
        
        Returns:
            list[Tensor]: A list containing the weight and bias tensors.
        """
        # parameters = [self.weight]
        # if self.bias is not None:   
        #     parameters.append(self.bias)
        # return parameters
        return super().parameters()
    
    def __repr__(self) -> str:
        """String representation of the linear layer."""
        bias_str = ", bias=True" if self.bias is not None else ""
        return (f"Linear(in_features={self.in_features}, "
                f"out_features={self.out_features}{bias_str})")
    
class Dropout(Layer):
    """Initialize dropout layer.
    Args:
        p (float): Probability of dropping a unit. Must be in the range [0.0, 1.0).
    """
    def __init__(self, p: float, training: bool=False) -> None:
        if not (DROPOUT_MIN_PROB <= p < DROPOUT_MAX_PROB):
            raise ValueError(f"Dropout probability must be in the range [{DROPOUT_MIN_PROB}, {DROPOUT_MAX_PROB}), got {p}.")
        self.p = p
        self.training = training
        
    def forward(self, input: Tensor) -> Tensor:
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
        if not self.training or self.p == DROPOUT_MIN_PROB:
            return input
        
        if self.p == DROPOUT_MAX_PROB:
            return Tensor(np.zeros_like(input.data), requires_grad=input.requires_grad)
        
        #* during training, apply dropout
        keep_prob = DROPOUT_MAX_PROB - self.p
        mask = (np.random.random(input.data.shape) < keep_prob)
        
        #* apply mask and scale to preserve gradients
        mask_tensor = Tensor(mask.astype(np.float32), requires_grad=False)
        scale = 1.0 / keep_prob
        output = (input * mask_tensor) * scale
        return output
    
    def __call__(self, input: Tensor) -> Tensor:
        return self.forward(input)
    
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
    def __init__(self, *layers: Tuple[Layer] | list[Layer]):
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
    
    def parameters(self) -> list[Parameter]:
        """Returns the parameters of all layers in the Sequential container."""
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params
        
    def __repr__(self) -> str:
        layers_repr = ",\n  ".join(repr(layer) for layer in self.layers)
        return f"Sequential(\n  {layers_repr}\n)"
    
    
class Residual(Layer):
    def __init__(self, fn: Layer) -> None:
        super().__init__()
        self.fn = fn
        
    def forward(self, X: Tensor) -> Tensor:
        return self.fn(X) + X
    
    def __repr__(self) -> str:
        return f"Residual({self.fn})"
    
    def __call__(self, input: Tensor) -> Tensor:
        return self.forward(input)
    

class LayerNormalization(Layer):
    """
    Normalizes across the last dimension of the input tensor
    """
    def __init__(self, num_features: int, eps: float= EPSILON) -> None:
        super().__init__()
        if not isinstance(num_features, int) or num_features <= 0:
            raise ValueError(
                f"num_features must be a positive integer, got {num_features}"
            )
            
        self.dim = num_features
        self.eps = eps
        self.weight = Parameter(Tensor(np.ones(shape=self.dim)))
        self.bias = Parameter(Tensor(np.zeros(shape=self.dim)))
        
    def forward(self, X: Tensor) -> Tensor:
        assert X.shape[-1] == self.dim,\
            f'last dim of input should equal dim, received: {X.shape[1]} expected: {self.dim}'
            
        mean = X.mean(axis=-1, keepdims=True)
        var = X.var(axis=-1, keepdims=True)
        norm = (X - mean) / ((var + self.eps) ** 0.5)
        shifted_norm = self.weight * norm + self.bias
        return shifted_norm
    
    def __call__(self, input: Tensor, ) -> Tensor:
        return self.forward(input)
    
class BatchNormalization(Layer):
    """
    Normalizes across the first dimension. 
    
    Supports 2D inputs of shape (batch_size, num_features)
    """
    def __init__(
        self,
        num_features: int,
        eps: float = EPSILON,
        momentum: float = 0.1
        ) -> None:
        
        super().__init__()
        if not isinstance(num_features, int) or num_features <= 0:
            raise ValueError(
                f"num_features must be a positive integer, got {num_features}"
            )
            
        self.dim = num_features
        self.eps = eps
        self.momentum = momentum
        
        #* learnable parameters
        self.weight = Parameter(Tensor(np.ones(shape=self.dim)))
        self.bias = Parameter(Tensor(np.zeros(shape=self.dim)))
        
        #* running mean and var (statistics not learnable)
        self.running_mean = Tensor(np.zeros(shape=self.dim))
        self.running_var = Tensor(np.ones(shape=self.dim))
    
    def forward(self, X: Tensor) -> Tensor:
        assert len(X.shape) == 2, \
            f'Batch Normalization only supports 2D inputs (N,C), got shape {X.shape}'
        
        if X.shape[1] != self.dim:
            raise ValueError(
                f"Expected: {self.dim} features, got: {X.shape[1]}"
            )
            
        if self.training:
            batch_mean = X.mean(axis=0, keepdims=True)
            batch_var = X.var(axis=0, keepdims=True)
            
            #* update running statistics
            self.running_mean = (
                (1 - self.momentum) * self.running_mean +
                self.momentum * batch_mean.detach()
            )
            
            self.running_var = (
                (1 - self.momentum) * self.running_var + 
                self.momentum * batch_var.detach()
            )
            mean = batch_mean
            var = batch_var
        else:
            mean = self.running_mean
            var = self.running_var
        
        x_hat = (X - mean) / ((var + self.eps) **0.5)
        out = self.weight * x_hat + self.bias
        return out
    
    
    
    
    
    
    
    
    
    
    
    
    

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