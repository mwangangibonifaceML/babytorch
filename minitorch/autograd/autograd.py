"""Autograd ⚡ - The Gradient Engine
**CRITICAL**: This module enhances the existing Tensor class - no new wrapper classes needed!
"""
from minitorch.tensor.tensor import Tensor
import numpy as np

EPSILON = 1e-7

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
    

class SubBackward(Function):
    """
    Gradient computation for subtraction

    Args:
        grad_output (Tensor): gradient flowing from the output (next layer).
        
    ** Key Insight **
        The gradient of subtraction is simply the gradient flowing from
        the output (next layer) mutliplied by eather 1 or -1 depending the sign of the operand.
        
        ** Mathematical Justification **
        If z = a - b, then: 
        
        ∂z/∂a = 1
        ∂z/∂b = -1
        Therefore, the gradients for both a becomes the gradient output from the next layer and
        gradient becomes negative of the gradient output from the next layer.

    """
    def __call__(self, grad_output: Tensor) -> tuple[Tensor, Tensor]:
        """Calculate gradient for matrix subtraction

        Args:
            grad_output (Tensor): gradient flowing from the output (next layer)

        Returns:
            tuple[Tensor, Tensor]: gradients for the first and second operands of the addition

        
        """
        a, b = self.saved_tensors
        assert isinstance(a, Tensor) and isinstance(b, Tensor), 'All inputs to Function must be instances of Tensor'

        grad_a = Tensor(np.zeros_like(a.data))
        grad_b = Tensor(np.zeros_like(b.data))
        
        if isinstance(a, Tensor) and a.requires_grad:
            grad_a = grad_output
        
        if isinstance(b, Tensor) and b.requires_grad:
            grad_output = -grad_output
            grad_b = grad_output
        
        return grad_a, grad_b
    
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
        assert isinstance(a, Tensor) and isinstance(b, Tensor),'All inputs to Function must be instances of Tensor'

        grad_a = Tensor(np.zeros_like(grad_output.data))
        grad_b = Tensor(np.zeros_like(grad_output.data))
        
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
        assert isinstance(a, Tensor) and isinstance(b, Tensor), 'All inputs to Function must be instances of Tensor'
        
        grad_a = Tensor(np.zeros_like(a.data))
        grad_b = Tensor(np.zeros_like(b.data))
        
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
        assert isinstance(a, Tensor) and isinstance(b, Tensor),'All inputs to Function must be instances of Tensor'
        
        grad_a = Tensor(np.zeros_like(a.data))
        grad_b = Tensor(np.zeros_like(b.data))
        
        if isinstance(a, Tensor) and a.requires_grad:
            grad_a = grad_output @ b.transpose()
    
        if isinstance(b, Tensor) and b.requires_grad:
            grad_b = a.transpose() @ grad_output
        
        return grad_a, grad_b
    
class DivBackward(Function):
    def __call__(self, grad_output: Tensor) -> tuple[Tensor, Tensor]:
        a,b = self.saved_tensors
        assert isinstance(a, Tensor) and isinstance(b, Tensor),'All inputs to Function must be instances of Tensor'
        
        grad_a = Tensor(np.zeros_like(a.data))
        grad_b = Tensor(np.zeros_like(b.data))
        
        grad_a = grad_output / b if a.requires_grad else Tensor(np.zeros_like(grad_output.data))
        grad_b = -grad_output * a / b ** 2 if b.requires_grad else Tensor(np.zeros_like(grad_output.data))
        
        return grad_a, grad_b
    
class TransposeBackward(Function):
    """
    Gradient computation for transpose operation.
    
    **Mathematical foundation**
    d(X.T)/d(X) = grad_output.T.
    Just transpose the gradient back
    
    **Mathematical Justification**
    The gradient of a transpose is just transpose of the gradient output.
    This is because transpose is a linear operation that just rearranges elements
    
    **Application**
    used in attention to find Key transpose, weight transpose, etc.
    """
    def __init__(self, tensor: Tensor,dim1:int, dim2:int) -> None:
        super().__init__(tensor)
        self.dim0 = dim1
        self.dim1 = dim2
        
    def __call__(self, grad_output: Tensor) -> tuple[Tensor,Tensor]:
        """
        calculate gradient for transpose operation
        
        """
        x, = self.saved_tensors
        grad_x = Tensor(np.zeros_like(grad_output.data))

        if isinstance(x, Tensor) and x.requires_grad:
            # Transpose gradient using the same dims
            if self.dim0 is None and self.dim1 is None:
                # Default: transpose last two dimensions
                if grad_output.data.ndim < 2:
                    grad_x = Tensor(grad_output.data.copy())
                else:
                    axes = list(range(grad_output.data.ndim))
                    axes[-2], axes[-1] = axes[-1], axes[-2]
                    grad_x = Tensor(np.transpose(grad_output.data, axes))
            else:
                # Specific dimensions: swap them back
                axes = list(range(grad_output.data.ndim))
                axes[self.dim0], axes[self.dim1] = axes[self.dim1], axes[self.dim0]
                grad_x = Tensor(np.transpose(grad_output.data, axes))
        return (grad_x,)
    
class PermuteBackward(Function):
    """
        Compute gradient for permutation.

        The gradient is permuted back using the inverse permutation.

        **Mathematical Foundation:**
            - ∂(X.permute(axes))/∂X = grad_output.permute(inverse_axes)
        
        **Example:**
            If axes = (0, 2, 1, 3), the inverse is (0, 2, 1, 3) (self-inverse).
            More generally, if axes = (2, 0, 1), the inverse is (1, 2, 0).  
        
        Args:
            tensor : Input tensor
            axes : tuple of axis indices defining the permutation order.
            
        Application:
            Multi-Head Attention use (0,2,1,3) to rearrange heads
        """
    def __init__(self, tensors: Tensor, axes: tuple[int,...]) -> None:
        super().__init__(tensors)
        self.axes = axes
        self.inverse_axes = tuple(np.argsort(axes)) #* if axes[i] = j, then inverse_axes[j] = i
        
    def __call__(self, grad_output: Tensor) -> Tensor:
        x, _ = self.saved_tensors
        grad_x = Tensor(np.zeros_like(grad_output.data))

        if isinstance(x, Tensor) and x.requires_grad:
            grad_x = Tensor(np.transpose(grad_output.data, self.inverse_axes))
        return (grad_x,)
        
    
class EmbeddingBackward(Function):
    """
    Gradient computation for embedding lookup.
    
    **Mathematical foundation**
        if Y = Embedding[indices], then
        - ∂Loss/∂Embedding[i] = sum of all gradients where index==i
        
    **Key insight**
        - Embedding lookup is a gather operation. The backward is a scatter operation
        that accumulates gradients to the embedding weights
        
    **Applications**
        - Word embedings, positional embeddings and token embeddings
    
    """
    def __init__(self, weight, indices) -> None:
        super().__init__(weight)
        self.indices = indices
        
    def __call__(self, grad_output: Tensor) -> Tensor:
        weight, _ = self.saved_tensors
        grad_weight = Tensor(np.zeros_like(weight.data))
        
        if isinstance(weight, Tensor) and weight.requires_grad:
            grad_weight = Tensor(np.zeros_like(weight.data))
            indices_flat = self.indices.data.dtype(int).flatten()
            grad_output_reshaped = grad_output.reshape(-1, grad_output.shape[-1])
            np.add.at(grad_weight.data, indices_flat, grad_output_reshaped.data)
            
        return (grad_weight,)
    
class SliceBackward(Function):
    """
    Gradient computation for tensor slicing/indexing operations.

    **Mathematical Rule:** If Y = X[key], then:
    - ∂Loss/∂X[key] = grad_output
    - ∂Loss/∂X[other positions] = 0

    **Key Insight:** Slicing is a masking operation. The backward
    places gradients back into the original tensor positions, with
    zeros everywhere else.

    **Applications:** Positional encodings, sequence slicing, batch selection,
    attention masking in transformers.

    **Examples:**
    >>> x = Tensor([1, 2, 3, 4, 5], requires_grad=True)
    >>> y = x[:3]  # Slice first 3 elements
    >>> loss = y.sum()
    >>> loss.backward()
    >>> # x.grad = [1, 1, 1, 0, 0] - gradients only for sliced positions
    """

    def __init__(self, tensor, key):
        """
        Args:
            tensor: Original tensor being sliced
            key: Slicing key (index, slice, tuple of slices, etc.)
        """
        super().__init__(tensor)
        self.key = key
        self.original_shape = tensor.shape

    def __call__(self, grad_output):
        """
        Compute gradient for slicing operation.

        Args:
            grad_output: Gradient flowing backward from sliced output

        Returns:
            Tuple with single gradient for input tensor

        **Mathematical Foundation:**
        - Slicing extracts a subset of elements
        - Backward scatters gradients back to original positions
        - Unsliced positions receive zero gradient

        **Example:**
        If X = [a, b, c, d, e] and Y = X[1:4] = [b, c, d]
        Then dL/dX = [0, dL/db, dL/dc, dL/dd, 0]

        TODO: Implement gradient computation for slicing/indexing operation.

        APPROACH:
        1. Extract input tensor from self.saved_tensors
        2. Initialize grad_input to None
        3. If tensor requires gradients:
            - Create zeros array: grad_input = np.zeros(self.original_shape)
            - Place gradients back: grad_input[self.key] = grad_output
        4. Return tuple (grad_input,)

        EXAMPLE:
        >>> X = Tensor([1, 2, 3, 4, 5], requires_grad=True)
        >>> Y = X[:3]  # Slice first 3 elements → [1, 2, 3]
        >>> # During backward: grad_output = [1, 1, 1]
        >>> # grad_X = [1, 1, 1, 0, 0] (gradients only for sliced positions)

        HINTS:
        - Create zero gradient array with original tensor shape
        - Use fancy indexing: grad_input[self.key] = grad_output
        - This automatically handles all slice types (single index, ranges, tuples)
        - Return as single-element tuple: (grad_input,)
        """
        tensor, = self.saved_tensors
        grad_input = Tensor(np.zeros_like(grad_output.data))

        if isinstance(tensor, Tensor) and tensor.requires_grad:
            # Create gradient array with same shape as original tensor
            grad_input = Tensor(np.zeros(self.original_shape, dtype=np.float32))

            # Place gradients back into the sliced positions
            # This is the inverse of the forward slicing operation
            grad_input[self.key] = grad_output

        return (grad_input,)

class ReshapeBackward(Function):
    """
    Gradient computation for reshape operation.

    **Mathematical Rule:** If Y = X.reshape(new_shape), then:
    - ∂Y/∂X = grad_Y.reshape(X.shape)

    **Key Insight:** Reshape just rearranges the same elements.
    The gradient is simply reshaped back to the original shape!

    **Applications:** Flattening tensors for linear layers, reshaping
    between convolutional and dense layers.
    """

    def __init__(self, tensor, original_shape):
        """
        Args:
            tensor: Input tensor
            original_shape: Shape before reshape
        """
        super().__init__(tensor)
        self.original_shape = original_shape

    def __call__(self, grad_output):
        """
        Compute gradient for reshape.

        Args:
            grad_output: Gradient flowing backward from output

        Returns:
            Tuple with single gradient for input tensor

        **Mathematical Foundation:**
        - ∂(X.reshape(...))/∂X = grad_output.reshape(X.shape)
        - Just reshape the gradient back!

        TODO: Implement gradient computation for reshape operation.

        APPROACH:
        1. Extract input tensor x from self.saved_tensors
        2. Initialize grad_x to None
        3. If x requires gradients:
            - Reshape grad_output back to original shape
            - Use grad_output.reshape(self.original_shape)
        4. Return tuple (grad_x,)

        EXAMPLE:
        >>> X = Tensor([[1, 2], [3, 4]], requires_grad=True)  # (2, 2)
        >>> Y = X.reshape(4)  # [1, 2, 3, 4]
        >>> # During backward: grad_output = [1, 1, 1, 1]
        >>> # grad_X = grad_output.reshape((2, 2)) = [[1, 1], [1, 1]]

        HINTS:
        - Reshape just rearranges elements, doesn't change values
        - Simply reshape gradient back to original shape
        - Use .reshape() method on grad_output numpy array
        - Return as single-element tuple: (grad_x,)
        """
        x, = self.saved_tensors
        grad_x = None

        if isinstance(x, Tensor) and x.requires_grad:
            # Reshape gradient back to original shape
            grad_x = Tensor(grad_output.reshape(self.original_shape))

        return (grad_x,)


class SumBackward(Function):
    """
    Gradient computation for tensor sum.

    **Mathematical Rule:** If z = sum(a), then ∂z/∂a[i] = 1 for all i

    **Key Insight:** Sum distributes the gradient equally to all input elements.
    The gradient is broadcast from the reduced output back to input shape.

    **Applications:** Used in loss functions, mean operations, and
    anywhere tensor reduction occurs.
    """

    def __call__(self, grad_output):
        """
        Compute gradients for sum operation.

        Args:
            grad_output: Gradient flowing backward from output

        Returns:
            Tuple containing gradient for the input tensor

        **Mathematical Foundation:**
        - ∂sum(a)/∂a[i] = 1 → grad_a = ones_like(a) * grad_output

        TODO: Implement gradient computation for sum reduction operation.

        APPROACH:
        1. Extract input tensor from self.saved_tensors
        2. If tensor requires gradients:
            - Create ones array: np.ones_like(tensor.data)
            - Multiply by grad_output: ones * grad_output
            - Return as tuple: (grad_tensor,)
        3. Else return (None,)

        EXAMPLE:
        >>> X = Tensor([1, 2, 3], requires_grad=True)
        >>> Y = X.sum()  # Y = 6 (scalar)
        >>> # During backward: grad_output = 1 (scalar)
        >>> # grad_X = [1, 1, 1] * 1 = [1, 1, 1]

        HINTS:
        - Sum distributes gradient equally to all elements
        - Use np.ones_like(tensor.data) to create gradient template
        - Multiply ones by grad_output (broadcasting handles scalar/tensor)
        - Return as single-element tuple: (grad_result,)
        """
        tensor, = self.saved_tensors

        if isinstance(tensor, Tensor) and tensor.requires_grad:
            return Tensor(np.ones_like(tensor.data)) * grad_output,
        return None,
        
        
class ReLUBackward(Function):
    """
    Gradient computation for ReLU activation
    
    **Mathematical foundation**
    ReLU: f(x) = max(0,x)
          f'(x) = 1 if x>0, else 0
    """
    def __init__(self, input_tensor):
        super().__init__(input_tensor)
        
    def __call__(self, grad_output):
        input_tensor, = self.saved_tensors
        gradient = Tensor(np.zeros_like(input_tensor.data))
        
        if isinstance(input_tensor, Tensor) and input_tensor.requires_grad:
            relu_gradient = (input_tensor.data > 0).dtype(np.float32)
            gradient = Tensor(grad_output.data * relu_gradient)
        return gradient
    
class SigmoidBackward(Function):
    """
    Gradient computation for sigmoid activation
    
    **Mathematical Intuation**
    Sigmoid: f(x) = 1/(1+exp(-x))
            f'(x) = f(x)*(1-f(x))
    
    """
    def __init__(self, input_tensor: Tensor, output_tensor: Tensor)-> None:
        super().__init__(input_tensor)
        self.output_tensor = output_tensor
        
    def __call__(self, grad_output: Tensor) -> Tensor:
        tensor, _ = self.saved_tensors
        gradient = Tensor(np.zeros_like(grad_output.data))
        
        if isinstance(tensor, Tensor) and tensor.requires_grad:
            sigmoid_grad = self.output_tensor * (1 - self.output_tensor)
            gradient = grad_output * sigmoid_grad
        return gradient
    
class SoftMaxBackward(Function):
    """
    Gradient computation for softmax activation
    
    **Mathematical Intuation**
    Softmax: f(x) = exp(x)/sum(exp(x))
    derivative = ∂softmax/∂x[i] = softmax[i] * (δ[i,j] - softmax[j])

    For gradient computation:
    grad_x[i] = softmax[i] * (grad_y[i] - sum(grad_y * softmax))

    
    """
    def __init__(self, input_tensor: Tensor, output_tensor: Tensor, dim=-1)-> None:
        super().__init__(input_tensor)
        self.output_tensor = output_tensor
        self.dim = dim
        
    def __call__(self, grad_output: Tensor)-> Tensor:
        tensor, _ = self.saved_tensors
        gradient = Tensor(np.zeros_like(grad_output.data))
        
        if isinstance(tensor, Tensor) and tensor.requires_grad:
            sum_term = Tensor.sum(grad_output * self.output_tensor,axis=self.dim, keepdims=True)
            gradient = self.output_tensor * (grad_output - sum_term)
        return gradient
    
class GeLUBackward(Function):
    def __init__(selfm input_tensor: Tensor)-> None:
        super().__init__(input_tensor)
        
    def __call__(self, grad_output: Tensor):
        tensor, = self.saved_tensors

        if isinstance(tensor, Tensor) and tensor.requires_grad:
            x = tensor.data
            # GELU derivative approximation
            # Using the tanh approximation:
            # gelu(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            sqrt_2_over_pi = np.sqrt(2.0 / np.pi)
            x_cubed = x ** 3
            tanh_arg = sqrt_2_over_pi * (x + 0.044715 * x_cubed)
            tanh_out = np.tanh(tanh_arg)
            sech_squared = 1 - tanh_out ** 2

            # Derivative: 0.5 * (1 + tanh(...)) + 0.5 * x * sech²(...) * d(tanh_arg)/dx
            d_tanh_arg = sqrt_2_over_pi * (1 + 0.134145 * x ** 2)
            gelu_grad = 0.5 * (1 + tanh_out) + 0.5 * x * sech_squared * d_tanh_arg

            return (grad_output * gelu_grad,)
        
class MSEBackward(Function):
    def __init__(self, predictions: Tensor, targets: Tensor)-> None:
        super().__init__(predictions)
        self.targets = targets
        self.num_samples = len(targets)
        
    def __call__(self, grad_output: Tensor) -> Tensor:
        predictions, _ = self.saved_tensors
        gradient = Tensor(np.zeros_like(grad_output.data))
        
        if isinstance(predictions, Tensor) and predictions.requires_grad:
            grad = 2.0 * (predictions - self.targets) / self.num_samples
            gradient = grad * grad_output
        return gradient
    
    
class BCEBackward(Function):
    def __init__(self, predictions: Tensor, targets: Tensor)->None:
        super().__init__(predictions)
        self.targets = targets
        self.num_samples = len(targets)
        
    def __call__(self, grad_output: Tensor)-> Tensor:
        predictions,_ = self.saved_tensors
        gradient = Tensor(np.zeros_like(grad_output.data))
        
        if isinstance(predictions, Tensor) and predictions.requires_grad:
            eps = EPSILON
            p = np.clip(predictions.data, eps, 1-eps)
            y = self.targets.data
            grad = (p - y) / (p* (1 - p)) * self.num_samples
            gradient = Tensor(grad * grad_output.data)
        return gradient
    
    
class CrossEntropyBackward(Function):
    def __init__(self, logits: Tensor, targets: Tensor)-> None:
        super().__init__(logits)
        self.targets = targets
        self.batch_size = logits.data.shape[0]
        self.num_classes = logits.data.shape[1]
        
    def __call__(self, grad_output: Tensor) -> Tensor:
        logits, _ = self.saved_tensors
        gradient = Tensor(np.zeros_like(grad_output.data))
        
        if isinstance(logits, Tensor) and logits.requires_grad:
            max_logits = np.max(logits.data, axis=1, keepdims=True)
            exp_logits = np.exp(logits.data - max_logits)
            softmax = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            
            one_hot = np.zeros((self.batch_size, self.num_classes), dtype=np.float32)
            one_hot[np.arange(self.batch_size), self.targets.data] = 1.0
            
            grad = (softmax - one_hot) / self.batch_size
            gradient = Tensor(grad * grad_output)
        return gradient