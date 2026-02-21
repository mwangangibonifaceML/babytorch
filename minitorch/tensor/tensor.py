#------------------------------------------------------------------------------------#
#* File: tensor.py - Implementation of a basic Tensor class using NumPy
#* A simple Tensor class that wraps a NumPy array and provides basic tensor operations.
#* This serves as the foundational data structure for building machine learning models.
#------------------------------------------------------------------------------------#

from __future__ import annotations
import numpy as np
from typing import Union, Optional, List, Any
from numpy.typing import NDArray

#* Constants for memory calculations
MB_TO_BYTES = 1024 * 1024
BYTES_PER_FLOAT32 = 4

def unbroadcast(grad, shape):
    """
    Sum grad so that it matches target_shape.
    """
    #* remove the leading dimension
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
        
    #* sum over the broadcasted dimension
    for i, dim in enumerate(shape):
        if dim == 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad

#* Basic Tensor class
class Tensor:
    """A tensor class that tries to mimick a pytorch tensor
    
    This class starts simple but includes dormant features for future modules:
    - requires_grad: Will be used for automatic differentiation (implemented in autograd module)
    - grad: Will store computed gradients (implemented in autograd module)
    - backward(): Will compute gradients (the core idea in autograd module)

    For now,this class focuses on: data, shape, and basic operations.
    """
    def __init__(
        self,
        data: Union[NDArray, List[int|float], int, float],
        requires_grad: bool =False,
        _parents=()
        ) -> None:
        # assert isinstance(data, (NDArray, float, int, Tensor))
        
        if isinstance(data, (float, int)):
            self.data = np.array(data)
            
        elif isinstance(data, Tensor):
            self.data = data.data
        else:
            self.data = data
        
        # if self.data.dtype == object:
        #     raise ValueError(
        #         f'Tensor cannot wrap object arrays.'
        #         'You likely created a NumPy array of Tensor objects.'
        #     )
            
        self.shape = self.data.shape        #* Tuple[int, ...], look at the shape attribute of numpy arrays
        self.size = self.data.size          #* Int, look at the number of elements in numpy arrays
        self.dtype = self.data.dtype        #* Data type of the underlying numpy array
        self.requires_grad = requires_grad  #* The trigger for gradient computation (dormant feature)
        
        self.grad = np.zeros_like(
            self.data, dtype=np.float32) if requires_grad else None      #* Placeholder: store gradients here in future modules
        
        self._backward = lambda: None
        self._parents = set(_parents)
        
    def __repr__(self) -> str:
        gradient_info = f"requires_grad={self.requires_grad}" if self.requires_grad else None
        return f"Tensor(data={self.data}, shape={self.shape}, grad_info= {gradient_info})"
    
    def __str__(self) -> str:
        return f"Tensor(data={self.data})"
    
    @staticmethod
    def __ensure_tensor(x: Union[int, float, np.ndarray]) -> Tensor:
        """Check whether an argument is a Tensor, if not
        wrap it in Tensor class
        """
        if not isinstance(x, Tensor):
            return Tensor(x)
        return x
    
    def numpy(self)-> np.ndarray:
        """Return the underlying numpy array"""
        return self.data.copy()
    
    def copy(self) -> Tensor:
        return Tensor(self.data.copy())
    
    def memory_footprint(self)-> int:
        """Calculate exact memory usage in bytes
        
        Returns:
            int: Memoery usage in bytes (eg., 1000X1000 float32 = 4MB)
        """
        return int(self.data.nbytes)
    
    def _determine_gradient_requirement(self, other: Any)-> bool:
        if isinstance(other, Tensor):
            return self.requires_grad or other.requires_grad
        return self.requires_grad
    
    def __len__(self)->int:
        return len(self.data)
    
    def __add__(self, other)-> Tensor:
        """Add two tensors element-wise with broadcasting support
        """
        other = Tensor.__ensure_tensor(other)
        result = Tensor(self.data + other.data,
                    requires_grad= self._determine_gradient_requirement(other),
                    _parents= (self, other))
        
        def _backward():
            if self.requires_grad:
                self._add_grad(unbroadcast(result.grad, self.shape))
                
            if other.requires_grad:
                other._add_grad(unbroadcast(result.grad, other.shape))
                
        result._backward = _backward
        return result
    
    def __radd__(self, other):
        return self.__add__(other)
    
        
    def __mul__(self, other)-> Tensor:
        """Multiply two tensors element-wise (NOT matrix multiplication)."""
        other = Tensor.__ensure_tensor(other)
        result = Tensor(
                    np.multiply(self.data , other.data),
                    requires_grad= self._determine_gradient_requirement(other),
                    _parents=(self,other))
        
        def _backward():
            if self.requires_grad:
                self._add_grad(other.data * result.grad)
                
            if other.requires_grad:
                other._add_grad(self.data * result.grad)
        
        result._backward = _backward
        return result
    
    def __rmul__(self, other)-> Tensor:
        return self.__mul__(other)
        
    def __sub__(self, other)-> Tensor:
        """Subtract two tensors element-wise."""
        if not isinstance(other, Tensor):
            other = Tensor(other)
        result = Tensor(self.data - other.data,
                    requires_grad= self._determine_gradient_requirement(other),
                    _parents=(self,other))
        
        def _backward():
            if self.requires_grad:
                self._add_grad(unbroadcast(result.grad, self.shape))
            
            if other.requires_grad:
                other._add_grad(-unbroadcast(result.grad, other.shape))
        
        result._backward = _backward
        return result
        
    def __truediv__(self, other)-> Tensor:
        """Divide two tensors element-wise."""
        if not isinstance(other, Tensor):
            other = Tensor(other)
        result = Tensor(self.data / other.data,
                        requires_grad= self._determine_gradient_requirement(other),
                        _parents=(self,other))
        
        def _backward():
            if self.requires_grad:
                self._add_grad(result.grad / other.data)
                
            if other.requires_grad:
                other._add_grad(-result.grad * self.data / other.data ** 2)  
        result._backward = _backward
        return result

    def __matmul__(self, other)-> Tensor:
        """Enable @ operator for matrix multiplication"""
        return self.matmul(other)
    
    def __neg__(self):
        result = Tensor(-self.data, requires_grad=self.requires_grad, _parents=(self,))
        
        def _backward():
            if self.requires_grad:
                self._add_grad(-result.grad)
        result._backward = _backward
        return result
    
    def __rsub__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        result = Tensor(other - self.data,
                        requires_grad=self._determine_gradient_requirement(other),
                        _parents=(self,))
        
        def _backward():
            if other.requires_grad:
                other._add_grad(result.grad)
            
            if self.requires_grad:
                self._add_grad(-result.grad)
        result._backward = _backward
        return result

    def __rtruediv__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        result = Tensor(other / self.data,
                        requires_grad=self._determine_gradient_requirement(other),
                        _parents=(self, other))
        
        def _backward():
            if other.requires_grad:
                other._add_grad(result.grad / self.data)
            
            if self.requires_grad:
                self._add_grad(-result.grad * other.data / self.data ** 2)
            
        result._backward = _backward
        return result
    
    def __pow__(self, other):
        if not isinstance(other, (float, int)):
            raise AssertionError('Power must be either integer or a float')
        result =  Tensor(self.data ** other,
                        requires_grad=self.requires_grad,
                        _parents=(self,))
        
        def _backward():
            if self.requires_grad:
                self._add_grad(other * (self.data ** (other - 1)) * result.grad)
        result._backward = _backward
        return result
    
    def matmul(self, other)-> Tensor:
        """Matrix multiplication for two tensors"""
        other = other if isinstance(other, Tensor) else Tensor(other)
        if self.shape == () or other.shape == ():
            result = Tensor(self.data * other.data, 
                        requires_grad= self._determine_gradient_requirement(other),
                        _parents=(self, other))
            
        if len(self.shape) == 0 or len(other.shape) == 0:
            result = Tensor(self.data * other.data,
                        requires_grad= self._determine_gradient_requirement(other),
                        _parents=(self,other))
            
        if len(self.shape) >= 2 and len(other.shape) >=2:
            if self.shape[-1] != other.shape[-2]:
                raise ValueError(
                    f"Cannot perform matrix multiplication: {self.shape} @ {other.shape}, "
                    f"Inner dimensions must match: {self.shape[-1] } ≠ {other.shape[-2]}"
                )
        
        #* Below piece of code will me slower than np.matmul
        #* Handle 2D|3D and so on matrices with explicit loops
        # start with 2D
        if len(self.shape) == 2 and len(other.shape) == 2:
            M,_ = self.data.shape
            _,N = other.data.shape
            results_data = np.zeros((M,N), dtype= self.dtype)
            result = Tensor(results_data,
                        requires_grad=self._determine_gradient_requirement(other),
                        _parents=(self,other))
            
            #* Explicit loops
            for row in range(M):
                for col in range(N):
                    results_data[row, col] = np.dot(self.data[row, :], other.data[:, col])
        else: # 3D+ operations
            results_data = np.matmul(self.data, other.data)
            
            result = Tensor(results_data,
                            requires_grad=self._determine_gradient_requirement(other),
                            _parents=(self,other))
        
        def _backward():
            if self.requires_grad:
                self._add_grad(result.grad @ other.data.T)
                
            if other.requires_grad:
                other._add_grad(self.data.T @ result.grad)
        
        result._backward = _backward
        return result
    
    def __getitem__(self, key):
        """Enable Tensor indexing and slicing"""
        assert key <= len(self.data), f'Index out of range, must be in range 0 - {len(self.data)}'
        results_data = self.data[key]

        if not isinstance(results_data, np.ndarray):
            results_data = np.array(results_data)
        result = Tensor(results_data, requires_grad= self.requires_grad, _parents=(self,))
        
        grad_input = np.zeros(self.data.shape, dtype=np.float32)
        grad_input[key] = result.grad
        
        def _backward():
            if self.requires_grad:
                self._add_grad(grad_input)
        result._backward = _backward
        
        return result
    
    
    def sum(self, axis=None, keepdims=False)-> Tensor:
        """Sum tensor along specified axis"""
        result =  Tensor(np.sum(self.data, axis=axis, keepdims= keepdims),
                    requires_grad= self.requires_grad, _parents=(self,))
        
        def _backward():
            if not self.requires_grad:
                return

            grad = result.grad

            # If dimensions were removed, restore them
            if axis is not None and not keepdims:
                if isinstance(axis, int):
                    axes = (axis,)
                else:
                    axes = axis

                for ax in sorted(axes):
                    grad = np.expand_dims(grad, ax)

            # Broadcast to input shape
            grad = np.broadcast_to(grad, self.shape)

            self._add_grad(grad)
        
        result._backward = _backward
        return result
    
    def mean(self, axis: int=None, keepdims: bool =False)-> Tensor:
        """Sum tensor along specified axis"""
        result =  Tensor(np.array(np.mean(self.data, axis=axis, keepdims= keepdims)),
                    requires_grad= self.requires_grad, _parents=(self,))
        
        def _backward():
            if self.requires_grad:
                self._add_grad(result.grad / self.data.size)
        result._backward = _backward
        return result
    
    def var(self, axis: int=None, keepdims: bool= False)-> Tensor:
        result = Tensor(
            np.array(np.var(self.data, axis=axis, keepdims=keepdims, ddof=1)),
            requires_grad=self.requires_grad, 
            _parents=(self,))
        
        def _backward():
            if self.requires_grad:
                self._add_grad(result.grad / self.data.size)
        result._backward = _backward
        return result      
    
    def max(self, axis:Optional[int]=None, keepdims=False) -> Tensor:
        """Find the max value of a tensor along specified axis"""
        return Tensor(np.array(np.max(self.data, axis=axis, keepdims=keepdims)),
                    requires_grad= self.requires_grad, _parents=(self,))
    
    def reshape(self, *shape):
        """Reshape the tensor to a new dimensions"""
        original_shape = self.shape
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            new_shape = tuple(shape[0])
        else:
            new_shape = shape
        if -1 in new_shape:
            if new_shape.count(-1) > 1:
                raise ValueError("Can only specify one unknown dimension with -1")
            known_size = 1
            unknown_idx = new_shape.index(-1)
            for i, dim in enumerate(new_shape):
                if i != unknown_idx:
                    known_size *= dim
            unknown_dim = self.size // known_size
            new_shape = list(new_shape)
            new_shape[unknown_idx] = unknown_dim
            new_shape = tuple(new_shape)
        if np.prod(new_shape) != self.size:
            target_size = int(np.prod(new_shape))
            raise ValueError(
                f"Total elements must match: {self.size} ≠ {target_size}"
            )
        reshaped_data = np.reshape(self.data, new_shape)
        result = Tensor(reshaped_data, requires_grad=self.requires_grad, _parents=(self,))
        
        def _backward():
            if self.requires_grad:
                self._add_grad(result.grad.reshape(original_shape))
        result._backward = _backward
        return result
    
    def transpose(self, dim0=None, dim1=None):
        """Transpose tensor dimensions."""
        ### BEGIN SOLUTION
        if dim0 is None and dim1 is None:
            if len(self.shape) < 2:
                transposed_data = self.data.copy()
            else:
                axes = list(range(len(self.shape)))
                axes[-2], axes[-1] = axes[-1], axes[-2]
                transposed_data = np.transpose(self.data, axes)
        else:
            axes = list(range(len(self.shape)))
            axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
            transposed_data = np.transpose(self.data, axes)
        result = Tensor(transposed_data, requires_grad=self.requires_grad, _parents=(self,))
        
        def _backward():
            if self.requires_grad:
                if dim0 is None and dim1 is None:
                    
                    grad =result.transpose()
                else:
                    grad = result.transpose(dim0, dim1)
        
        result._backward = _backward
        return result
    
    def detach(self):
        """Return a new Tensor detached from the current computation graph."""
        return Tensor(self.data.copy(), requires_grad=False)
    
    def backward(self):
        """Compute gradient"""
        topo =[]
        visited = set()
        def build_graph(root):
            if not root in visited:
                visited.add(root)
                for parent in root._parents:
                    build_graph(parent)
            topo.append(root)
            
        build_graph(self)
            
        self.grad = np.ones_like(self.data, dtype=np.float32)
        for node in reversed(topo):
            node._backward()
            
    def _add_grad(self, grad):
        if not self.requires_grad:
            return 
        
        if self.grad is None:
            self.grad = np.zeros_like(self.data, dtype=np.float32)
            
        # print(self.grad.shape, grad.shape)
        self.grad += grad
        
    #* ============ ACTIVATIONS ===============
    
    def tanh(self):
        "Hyperbolic tangent activation function"
        #* forward pass : exp(x) - exp(-x) / exp(x) + exp(-x)
        t = np.tanh(self.data)
        result = Tensor(t, requires_grad=self.requires_grad, _parents=(self,))
        
        def _backward():
            if self.requires_grad:
                self._add_grad((1 - t ** 2) * result.grad)
        result._backward = _backward
        return result
    
    def sigmoid(self):
        """Sigmoid activation: σ(x) = 1/(1 + e^(-x))
    
        Maps input to the range (0, 1).
        Perfect for probabilities and binary classification
        
        Args:
            X (Tensor): Input tensor.
        
        Returns:
            Tensor: Output tensor after applying Sigmoid activation.
        """
        #* Apply sigmoid: 1 / (1 + (-X).exp())
        #* clip extreme values to prevent overflows (sigmoid(-500) ~ 0, sigmoid(500) ~ 1)
        #* clipping at -500 and 500 ensures that exp(-x) stays within float64 range
        
        clipped_X = np.clip(self.data, -500, 500)
        
        #* Use numerical stable sigmoid formula
        #* for positive numbers: 1 / (1 + exp(-x))
        #* for negative numbers: exp(x) / (1 + exp(x))
        
        results = np.zeros_like(clipped_X)
        
        #* For positive numbers
        positive_mask = clipped_X >= 0
        z = np.exp(-clipped_X[positive_mask])
        results[positive_mask] = 1 / (1 + z)
        
        #* For negative numbers
        negative_mask = clipped_X < 0
        z = np.exp(clipped_X[negative_mask])
        results[negative_mask] = z / (1 + z)
        
        results = Tensor(results, requires_grad=self.requires_grad, _parents=(self,))
        
        if self.requires_grad:
            def _backward():
            
                sigmoid_gradient = results.data * (1 - results.data)
                self._add_grad(results.grad * sigmoid_gradient)
            results._backward = _backward
        return results