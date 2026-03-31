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
    
    This class starts simple but includes features for future modules:
    - device: To support CPU/GPU tensors in the future.
    - _parents: To track the computational graph for autograd.
    - data: The underlying NumPy array storing the tensor values.
    - requires_grad: Will be used for automatic differentiation.
    - grad: Will store computed gradients. 
    - backward(): Will compute gradients (the core idea in autograd).

    This class focuses on: data, shape, and basic operations.
    """
    def __init__(self, data, requires_grad=False, dtype='float32', device=None, _parents= tuple()):
        if isinstance(data, Tensor):
            if dtype is None:
                dtype = data.dtype
            self.data = data.data.astype(dtype)
            self.requires_grad = data.requires_grad
            self.device = data.device
            self._parents = data._parents
        elif isinstance(data, np.ndarray):
            self.data = data.astype(dtype)
        else:
            self.data = np.array(data, dtype=dtype)
        
        self.requires_grad = requires_grad    
        self.device = device if device is not None else 'cpu'
        self._parents = _parents
        self.grad = np.zeros(self.data.shape, dtype='float32')
        self._backward = lambda: None
        
    def __repr__(self) -> str:
        return f"Tensor(data={self.data}, shape={self.shape}, grad_info= {self.requires_grad})"
    
    def __str__(self) -> str:
        return f"Tensor(data={self.data})"
    
    @property
    def shape(self):
        return self.data.shape
    
    @property
    def size(self):
        return self.data.size
    
    @property
    def dtype(self):
        return self.data.dtype
    
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
                    _parents= (self, other),
                    dtype=self.dtype,
                    device=self.device)
        
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
                    _parents=(self,other),
                    dtype=self.dtype,
                    device=self.device)
        
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
                    _parents=(self,other),
                    dtype=self.dtype,
                    device=self.device)
        
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
                        _parents=(self,other),
                        dtype=self.dtype,
                        device=self.device)
        
        
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
        result = Tensor(-self.data,
                        requires_grad=self.requires_grad,
                        _parents=(self,),
                        dtype=self.dtype,
                        device=self.device)
        
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
                        _parents=(self,),
                        dtype=self.dtype,
                        device=self.device)
        
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
                        _parents=(self, other),
                        dtype=self.dtype,
                        device=self.device)
        
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
                        _parents=(self,),
                        dtype=self.dtype,
                        device=self.device)
        
        def _backward():
            if self.requires_grad:
                self._add_grad(other * (self.data ** (other - 1)) * result.grad)
        result._backward = _backward
        return result
    

    def matmul(self, other) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)

        if self.data.ndim == 0 or other.data.ndim == 0:
            # scalar multiplication fallback
            result = Tensor(
                self.data * other.data,
                requires_grad=self._determine_gradient_requirement(other),
                _parents=(self, other),
                dtype=self.dtype,
                device=self.device
            )
            
        else:
            if len(self.shape) >= 2 and len(other.shape) >=2:
                if self.shape[-1] != other.shape[-2]:
                    raise ValueError(
                        f"Incompatible shapes for matmul: "
                        f"{self.shape} @ {other.shape}"
                    )

            result_data = np.matmul(self.data, other.data)

            result = Tensor(
                result_data,
                requires_grad=self._determine_gradient_requirement(other),
                _parents=(self, other),
                dtype=self.dtype,
                device=self.device
            )

        def _backward():
            if result.grad is None:
                return

            grad_output = result.grad

            # Case 1: Matrix @ Vector
            if self.data.ndim == 2 and other.data.ndim == 1:
                # (m,k) @ (k,) -> (m,)
                if self.requires_grad:
                    # dX = outer(grad_output, w)
                    grad_self = np.outer(grad_output, other.data)
                    self._add_grad(grad_self)

                if other.requires_grad:
                    # dw = X^T @ grad_output
                    grad_other = self.data.T @ grad_output
                    other._add_grad(grad_other)
                    
            # Case 2: Vector @ Matrix
            else:
                if self.requires_grad:
                    grad_self = np.matmul(
                        grad_output,
                        np.swapaxes(other.data, -1, -2)
                    )
                    self._add_grad(grad_self)

                if other.requires_grad:
                    grad_other = np.matmul(
                        np.swapaxes(self.data, -1, -2),
                        grad_output
                    )
                    other._add_grad(grad_other)

        result._backward = _backward
        return result
    
    def __getitem__(self, key):
        """
        Enable Tensor indexing and slicing.
        Supports:
            tensor[i]
            tensor[i:j]
            tensor[:, 1:-1]
            tensor[:, -1]
            tensor[i, j]
        """

        # forward pass
        result_data = self.data[key]

        if not isinstance(result_data, np.ndarray):
            result_data = np.array(result_data)

        result = Tensor(
            result_data,
            requires_grad=self.requires_grad,
            _parents=(self,),
            dtype=self.dtype,
            device=self.device
        )

        def _backward():
            if not self.requires_grad:
                return

            grad_input = np.zeros_like(self.data)

            # Scatter gradient back to the indexed positions
            grad_input[key] += result.grad

            self._add_grad(grad_input)

        result._backward = _backward

        return result
    
    def sum(self, axis=None, keepdims=False)-> Tensor:
        """Sum tensor along specified axis"""
        result =  Tensor(np.sum(self.data, axis=axis, keepdims= keepdims),
                    requires_grad= self.requires_grad,
                    _parents=(self,),
                    dtype=self.dtype,
                    device=self.device)
        
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
                    requires_grad= self.requires_grad,
                    _parents=(self,),
                    dtype=self.dtype,
                    device=self.device)
        
        def _backward():
            if self.requires_grad:
                self._add_grad(result.grad / self.data.size)
        result._backward = _backward
        return result
    
    def var(self, axis: int=None, keepdims: bool= False)-> Tensor:
        result = Tensor(
            np.array(np.var(self.data, axis=axis, keepdims=keepdims, ddof=1)),
            requires_grad=self.requires_grad, 
            _parents=(self,),
            dtype=self.dtype,
            device=self.device)
        
        def _backward():
            if self.requires_grad:
                self._add_grad(result.grad / self.data.size)
        result._backward = _backward
        return result      
    
    def max(self, axis:Optional[int]=None, keepdims=False) -> Tensor:
        """Find the max value of a tensor along specified axis"""
        return Tensor(np.array(np.max(self.data, axis=axis, keepdims=keepdims)),
                    requires_grad= self.requires_grad, 
                    _parents=(self,),
                    dtype=self.dtype,
                    device=self.device)
    
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
        result = Tensor(reshaped_data, 
                        requires_grad=self.requires_grad,
                        _parents=(self,),
                        dtype=self.dtype,
                        device=self.device)
        
        def _backward():
            if self.requires_grad:
                self._add_grad(result.grad.reshape(original_shape))
        result._backward = _backward
        return result
    
    def transpose(self, dim0=None, dim1=None):
        """Transpose tensor dimensions."""
        ### BEGIN SOLUTION
        axis = None
        if dim0 is None and dim1 is None:
            if len(self.shape) < 2:
                transposed_data = self.data.copy()
            else:
                axes = list(range(len(self.shape)))
                axes[-2], axes[-1] = axes[-1], axes[-2]
                transposed_data = np.transpose(self.data, axes)
        else:
            if dim0 is None or dim1 is None:
                raise ValueError('Both dim1 and dim1 must be provided')
            if dim0 < 0 or dim1 < 0:
                raise ValueError('Dimensions must be non-negative')
            if dim0 >= len(self.shape) or dim1 >= len(self.shape):
                raise ValueError('Dimensions must be within the tensor shape')
            if dim0 == dim1:
                raise ValueError('Dimensions must be different')
            
            axes = list(range(len(self.shape)))
            axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
            axis = axes
            transposed_data = np.transpose(self.data, axes)
            
        result = Tensor(transposed_data,
                        requires_grad=self.requires_grad,
                        _parents=(self,),
                        dtype=self.dtype,
                        device=self.device)
        
        def _backward():
            if not self.requires_grad:
                return
            if result.grad is None:
                return 
                
            # transpose gradient back
            grad_input = np.transpose(result.grad, axis)

            if self.grad is None:
                self.grad = grad_input
            else:
                self._add_grad(grad_input)
        
        result._backward = _backward
        return result
    
    def detach(self):
        """Return a new Tensor detached from the current computation graph."""
        return Tensor(self.data.copy(), requires_grad=False)
    
    def backward(self):
        """Compute gradient"""
        #* build an ordered list of all tensors involved(Topological order)
        #* this ensures that when we call backward, we compute gradients in the correct order
        topo =[]
        visited = set()
        def build_graph(root):
            if not root in visited:
                visited.add(root)
                #* recursively visit parents first
                for parent in root._parents:
                    build_graph(parent)
            
                #* add self after all parents are visited
                topo.append(root)
            
        build_graph(self)
            
        #* initialize the gradient of the output tensor (self) to 1,
        #* since d(output)/d(output) = 1
        self.grad = np.ones_like(self.data, dtype=np.float32)
        
        #* go through the topology in reverse order triggering
        #* the backward function of each node
        for node in reversed(topo):
            node._backward()
            
    def _add_grad(self, grad: Tensor)->Tensor:
        """Accumulate the gradients"""
        if not self.requires_grad:
            return 
        
        if isinstance(grad, Tensor):
            grad = grad.data
        
        if self.grad is None:
            self.grad = np.zeros_like(self.data, dtype=np.float32)
            
        grad = unbroadcast(grad, self.shape)
        self.grad += grad
        
    @classmethod
    def from_numpy(cls,
                array: NDArray,
                requires_grad=False,
                dtype='float32', device=None) -> Tensor:
        """Create a Tensor from a NumPy array."""
        return cls(data=array, requires_grad=requires_grad, dtype=dtype, device=device)
        
    @classmethod
    def zeros(cls, shape, requires_grad=False, dtype='float32', device=None) -> Tensor:
        """Create a Tensor filled with zeros."""
        data = np.zeros(shape, dtype=dtype)
        return cls(data=data, requires_grad=requires_grad, dtype=dtype, device=device)
    
    @classmethod
    def ones(cls, shape, requires_grad=False, dtype='float32', device=None) -> Tensor:
        """Create a Tensor filled with ones."""
        data = np.ones(shape, dtype=dtype)
        return cls(data=data, requires_grad=requires_grad, dtype=dtype, device=device)
    
    @classmethod
    def randn(cls, shape, requires_grad=False, dtype='float32', device=None) -> Tensor:
        """Create a Tensor filled with random values from a normal distribution."""
        data = np.random.randn(*shape).astype(dtype)
        return cls(data=data, requires_grad=requires_grad, dtype=dtype, device=device)
    
    @classmethod
    def rand(cls, shape, requires_grad=False, dtype='float32', device=None) -> Tensor:
        """Create a Tensor filled with random values from a uniform distribution."""
        data = np.random.rand(*shape).astype(dtype)
        return cls(data=data, requires_grad=requires_grad, dtype=dtype, device=device)

    @classmethod
    def arange(cls, start, end=None, step=1, requires_grad=False, dtype='float32', device=None) -> Tensor:
        """Create a Tensor with values from start to end with a given step."""
        if end is None:
            end = start
            start = 0
        data = np.arange(start, end, step, dtype=dtype)
        return cls(data=data, requires_grad=requires_grad, dtype=dtype, device=device)
    
    @classmethod
    def constant(cls, value, shape=(), requires_grad=False, dtype='float32', device=None) -> Tensor:
        """Create a Tensor filled with a constant value."""
        data = np.full(shape, value, dtype=dtype)
        return cls(data=data, requires_grad=requires_grad, dtype=dtype, device=device)