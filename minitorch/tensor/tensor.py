#------------------------------------------------------------------------------------#
#* Module 01: Tensor Foundation - Building Blocks of ML
#* File: tensor.py - Implementation of a basic Tensor class using NumPy
#* A simple Tensor class that wraps a NumPy array and provides basic tensor operations.
#* This serves as the foundational data structure for building machine learning models.
#------------------------------------------------------------------------------------#

from __future__ import annotations
import numpy as np
from typing import Tuple, Union, Optional, List, Any, Iterable
from numpy.typing import NDArray

#* Constants for memory calculations
MB_TO_BYTES = 1024 * 1024
BYTES_PER_FLOAT32 = 4

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
        data: Union[NDArray, list[int|float], int, float],
        requires_grad: bool =False) -> None:
        if not isinstance(data, np.ndarray):
            self.data = np.array(data)
        else:
            self.data = data
            
        self.shape = self.data.shape        #* Tuple[int, ...], look at the shape attribute of numpy arrays
        self.size = self.data.size          #* Int, look at the number of elements in numpy arrays
        self.dtype = self.data.dtype        #* Data type of the underlying numpy array
        self.requires_grad = requires_grad  #* The trigger for gradient computation (dormant feature)
        self.grad = None                    #* Placeholder: store gradients here in future modules
        
    def __repr__(self) -> str:
        gradient_info = f"requires_grad={self.requires_grad}" if self.requires_grad else None
        return f"Tensor(data={self.data}, shape={self.shape}, grad_info= {gradient_info})"
    
    def __str__(self) -> str:
        return f"Tensor(data={self.data})"
    
    def numpy(self)-> np.ndarray:
        """Return the underlying numpy array"""
        return self.data.copy()
    
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
    
    def __add__(self, other)-> Tensor:
        """Add two tensors element-wise with broadcasting support
        """
        if isinstance(other, Tensor):
            return Tensor(self.data + other.data,
                        requires_grad= self._determine_gradient_requirement(other))
        else:
            return Tensor(self.data + other,
                        requires_grad= self._determine_gradient_requirement(other))
        
    def __mul__(self, other)-> Tensor:
        """Multiply two tensors element-wise (NOT matrix multiplication)."""
        if isinstance(other, Tensor):
            return Tensor(self.data * other.data,
                        requires_grad= self._determine_gradient_requirement(other))
        else:
            return Tensor(self.data * other,
                        requires_grad= self._determine_gradient_requirement(other))
        
    def __sub__(self, other)-> Tensor:
        """Subtract two tensors element-wise."""
        if isinstance(other, Tensor):
            return Tensor(self.data - other.data,
                        requires_grad= self._determine_gradient_requirement(other))
        else:
            return Tensor(self.data - other,
                        requires_grad= self._determine_gradient_requirement(other))
        
    def __truediv__(self, other)-> Tensor:
        """Divide two tensors element-wise."""
        if isinstance(other, Tensor):
            return Tensor(self.data / other.data,
                        requires_grad= self._determine_gradient_requirement(other))
        else:
            return Tensor(self.data / other,
                        requires_grad= self._determine_gradient_requirement(other))

    def __matmul__(self, other)-> Tensor:
        """Enable @ operator for matrix multiplication"""
        return self.matmul(other)
    
    def __rsub__(self, other):
        return Tensor(other - self.data, requires_grad=self.requires_grad)

    def __rtruediv__(self, other):
        return Tensor(other / self.data, requires_grad=self.requires_grad)
    
    def __pow__(self, other):
        if not isinstance(other, (float, int)):
            raise AssertionError('Power must be either integer or a float')
        return Tensor(self.data ** other, requires_grad=self.requires_grad)
    
    def matmul(self, other)-> Tensor:
        """Matrix multiplication for two tensors"""
        assert isinstance(other, Tensor), "Operand must be a Tensor"
        if self.shape == () or other.shape == ():
            return Tensor(self.data * other.data)
        if len(self.shape) == 0 or len(other.shape) == 0:
            return Tensor(self.data * other.data)
        if len(self.shape) >= 2 and len(other.shape) >=2:
            if self.shape[-1] != other.shape[-2]:
                raise ValueError(
                    f"Cannot perform matrix multiplication: {self.shape} @ {other.shape}, "
                    f"Inner dimensions must match: {self.shape[-1] } ≠ {other.shape[-2]}"
                )
        
        #* Below piece of code will me slower than np.matmul to demonstrate the values of
        #* vectorization. Will be accelerated later on
        
        #* Handle 2D|3D and so on matrices with explicit loops
        # start with 2D
        if len(self.shape) == 2 and len(other.shape) == 2:
            M,K = self.data.shape
            k2,N = other.data.shape
            results_data = np.zeros((M,N), dtype= self.dtype)
            
            #* Explicit loops
            for row in range(M):
                for col in range(N):
                    results_data[row, col] = np.dot(self.data[row, :], other.data[:, col])
        else: # 3D+ operations
            results_data = np.matmul(self.data, other.data)
        return Tensor(results_data, requires_grad=self._determine_gradient_requirement(other))
    
    def __getitem__(self, key):
        """Enable Tensor indexing and slicing"""
        assert key <= len(self.data), f'Index out of range, must be in range 0 - {len(self.data)}'
        results_data = self.data[key]
        if not isinstance(results_data, np.ndarray):
            results_data = np.array(results_data)
        return Tensor(results_data, requires_grad= self.requires_grad)
    
    def sum(self, axis=None, keepdims=False)-> Tensor:
        """Sum tensor along specified axis"""
        return Tensor(np.array(np.sum(self.data, axis=axis, keepdims= keepdims)),
                    requires_grad= self.requires_grad)
    
    def mean(self, axis=None, keepdims=False)-> Tensor:
        """Sum tensor along specified axis"""
        return Tensor(np.array(np.mean(self.data, axis=axis, keepdims= keepdims)),
                    requires_grad= self.requires_grad)

    def max(self, axis:Optional[int]=None, keepdims=False) -> Tensor:
        """Find the max value of a tensor along specified axis"""
        return Tensor(np.array(np.max(self.data, axis=axis, keepdims=keepdims)),
                    requires_grad= self.requires_grad)
    
    def backward(self):
        """Compute gradient"""
        #* implemented later
        pass
    
    def reshape(self, *shape):
        """Reshape the tensor to a new dimensions"""
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
        result = Tensor(reshaped_data, requires_grad=self.requires_grad)
        return result
    
    def transpose(self, dim0=None, dim1=None):
        """Transpose tensor dimensions."""
        ### BEGIN SOLUTION
        if dim0 is None and dim1 is None:
            if len(self.shape) < 2:
                return Tensor(self.data.copy())
            else:
                axes = list(range(len(self.shape)))
                axes[-2], axes[-1] = axes[-1], axes[-2]
                transposed_data = np.transpose(self.data, axes)
        else:
            if dim0 is None or dim1 is None:
                raise ValueError("Both dim0 and dim1 must be specified")
            axes = list(range(len(self.shape)))
            axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
            transposed_data = np.transpose(self.data, axes)
        result = Tensor(transposed_data, requires_grad=self.requires_grad)
        return result
    
    def detach(self):
        """Return a new Tensor detached from the current computation graph."""
        return Tensor(self.data.copy(), requires_grad=False)
