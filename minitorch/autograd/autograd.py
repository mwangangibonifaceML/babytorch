"""Autograd ⚡ - The Gradient Engine
**CRITICAL**: This module enhances the existing Tensor class - no new wrapper classes needed!
"""
from minitorch.tensor.tensor import Tensor
import numpy as np

class Function:
    def __init__(self, *tensors: Tensor) -> None:
        self.saved_tensors = tensors
        self.next_functions = []
        
        for tensor in tensors:
            if isinstance(tensor, Tensor) and tensor.requires_grad:
                if getattr(tensor, 'grad_fn', None) is not None:
                    self.next_functions.append(tensor.grad_fn)
                    
    def __call__(self, grad_output: Tensor)-> tuple[Tensor, Tensor]:
        assert isinstance(grad_output, Tensor), "grad_output must be a Tensor"
        raise NotImplementedError("Function __call__ must be implemented in subclasses")
    
class AddBackward(Function):
    """
    Gradient computation for addition

    Args:
        grad_output (Tensor): gradient flowing from the output (next layer).
        
    ** Key Insight **
        The gradient of addition is simply the gradient flowing from
        the output (next layer) to both operands since addition distributes the gradient equally.
        
        ** Mathematical Justification **
        If z = a + b, then: 
        
        ∂z/∂a = ∂z/∂b = ∂z/∂z = 1
        Therefore, the gradients for both a and b are equal to the incoming gradient grad_output.

    """
    def __call__(self, grad_output: Tensor) -> tuple[Tensor, Tensor]:
        """Calculate gradient for matrix addition

        Args:
            grad_output (Tensor): gradient flowing from the output (next layer)

        Returns:
            tuple[Tensor, Tensor]: gradients for the first and second operands of the addition

        
        """
        a, b = self.saved_tensors
        if not isinstance(a, Tensor) or not isinstance(b, Tensor):
            raise AssertionError('All inputs to Function must be instances of Tensor')
        assert isinstance(grad_output, Tensor), "grad_output must be a Tensor"

        # grad_a = grad_b = None
        
        if isinstance(a, Tensor) and a.requires_grad:
            grad_a = grad_output
        
        if isinstance(b, Tensor) and b.requires_grad:
            grad_b = grad_output
        
        return grad_a, grad_b
    
class MulBackward(Function):
    """
    Gradient computation for multiplication

    Args:
        grad_output (Tensor): gradient flowing from the output (next layer).
        
    ** Key Insight **
        The gradient of multiplication follows the product rule. The gradient 
        output is scaled by the value of the other operand.
        
    ** Mathematical Justification **
        If z = a * b, then: 
        
        ∂z/∂a = b
        ∂z/∂b = a
        Therefore, the gradients for a and b are scaled by the other operand.
        
    ** Application **
        This is crucial in weight scaling, attention mechanisms, and other neural network architectures
        where there is element wise multiplication involved.

    """
    def __call__(self, grad_output: Tensor) -> tuple[Tensor, Tensor]:
        """Calculate gradient for element-wise multiplication

        Args:
            grad_output (Tensor): gradient flowing from the output (next layer)
        
        Returns:
            tuple[Tensor, Tensor]: gradients for the first and second operands of the multiplication

        """
        # assert isinstance(grad_output, Tensor), "grad_output must be a Tensor"
        a, b = self.saved_tensors
        if not isinstance(a, Tensor) or not isinstance(b, Tensor):
            raise AssertionError('All inputs to Function must be instances of Tensor')
        assert isinstance(grad_output, Tensor), "grad_output must be a Tensor"
        # grad_b = grad_a = None
        
        if isinstance(a, Tensor) and a.requires_grad:
            if isinstance(b, Tensor):
                grad_a = grad_output * b.data
            else:
                grad_a = grad_output * b
    
        if isinstance(b, Tensor) and b.requires_grad:
            if isinstance(a, Tensor):
                grad_b = grad_output * a.data
            else:
                grad_b = grad_output * a
        
        return grad_a, grad_b
    
class MatMulBackward(Function):
    """
    Gradient computation for matrix multiplication

    Args:
        grad_output (Tensor): gradient flowing from the output (next layer).
        
    ** Key Insight **
        The gradient of matrix multiplication follows the rules of linear algebra.
        The gradient output is multiplied by the transpose of the other operand.
        
    ** Mathematical Justification **
        If z = A * B, then:
        
        ∂z/∂A = Bᵀ
        ∂z/∂B = Aᵀ
        Therefore, the gradients for A and B are scaled by the transpose of the other operand.
    """
    def __call__(self, grad_output: Tensor) -> tuple[Tensor, Tensor]:
        """Calculate gradient for matrix multiplication

        Args:
            grad_output (Tensor): gradient flowing from the output (next layer)
        
        Returns:
            tuple (Tensor, Tensor): gradients for the first and second operands of the matrix multiplication
            scaled by the gradient output.
        """
        a,b = self.saved_tensors
        if not isinstance(a, Tensor) or not isinstance(b, Tensor):
            raise AssertionError('All inputs to Function must be instances of Tensor')
        assert isinstance(grad_output, Tensor), "grad_output must be a Tensor"
        # grad_a = grad_b = None
        
        if isinstance(a, Tensor) and a.requires_grad:
            grad_a = grad_output @ b.transpose()
    
        if isinstance(b, Tensor) and b.requires_grad:
            grad_b = a.transpose() @ grad_output
        
        return grad_a, grad_b
    
class DivBackward(Function):
    def __call__(self, grad_output: Tensor) -> tuple[Tensor, Tensor]:
        a,b = self.saved_tensors
        if not isinstance(a, Tensor) or not isinstance(b, Tensor):
            raise AssertionError('All inputs to Function must be instances of Tensor')
        assert isinstance(grad_output, Tensor), "grad_output must be a Tensor"
        
        grad_a = grad_output / b if a.requires_grad else None
        grad_b = -grad_output * a / (b.__pow__(2)) if b.requires_grad else None
        return grad_a, grad_b