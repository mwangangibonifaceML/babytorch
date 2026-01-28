
import numpy as np
from typing import Optional, Tuple
from minitorch.tensor.tensor import Tensor

TOLERANCE = 1e-6

class Sigmoid:
    """Sigmoid activation: σ(x) = 1/(1 + e^(-x))
    
    Maps input to the range (0, 1).
    Perfect for probabilities and binary classification
    
    """
    def parameters(self):
        """Return empty list (activations have no learnable parameters)"""
        return []
    
    def forward(self, X: Tensor) -> Tensor:
        """Forward pass of the Sigmoid activation function.
        
        Approach:
        1. Apply sigmoid formula: σ(x) = 1/(1 + e^(-x))
        2. use np.exp for element-wise exponentiation.
        3. Return the result wrapped in new tensor.
        
        Example:
        >>> sigmoid = Sigmoid()
        >>> x = Tensor(np.array([-1.0, 0.0, 1.0]))
        >>> y = sigmoid(x)
        >>> print(y)
        Tensor([0.26894142, 0.5       , 0.73105858])
        
        HINT: Use np.exp(-X.data) for numerical stability.
        
        Args:
            X (Tensor): Input tensor.
        
        Returns:
            Tensor: Output tensor after applying Sigmoid activation.
        """
        #* Apply sigmoid: 1 / (1 + (-X).exp())
        #* clip extreme values to prevent overflows (sigmoid(-500) ~ 0, sigmoid(500) ~ 1)
        #* clipping at -500 and 500 ensures that exp(-x) stays within float64 range
        clipped_X = np.clip(X.data, -500, 500)
        
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
        return Tensor(results)
        
    def __call__(self, x: Tensor) -> Tensor:
        """Allows the activation to be called like a function."""
        return self.forward(x)

    # def backward(self, grad: Tensor) -> Tensor:
    #     """Backward pass of the Sigmoid activation function."""
    #     pass


class ReLU:
    
    """ReLU activation: f(x) = max(0, x)

    Sets negative values to zero, keeps positive values unchanged."""
    
    def parameters(self):
        """Return empty list (activations have no learnable parameters)"""
        return []
    
    def forward(self, X:Tensor) -> Tensor:
        """Forward pass of the ReLU activation function.
        
        Approach:
        1. use np.maximum(0, x.data) for element-wise max.
        2. Return the result wrapped in new tensor.
        
        Example:
        >>> relu = ReLu()
        >>> x = Tensor(np.array([-1.0, 0.0, 1.0]))
        >>> y = relu(x)
        >>> print(y)
        Tensor([0.0, 0.0, 1.0])
        
        Args:
            X (Tensor): Input tensor.
        
        Returns:
            Tensor: Output tensor after applying ReLU activation.
        """
        result = np.maximum(0, X.data)
        return Tensor(result)
    
    def __call__(self, x: Tensor) -> Tensor:
        """Allows the activation to be called like a function."""
        return self.forward(x)

    # def backward(self, grad: Tensor) -> Tensor:
    #     """Backward pass of the ReLU activation function."""
    #     pass

class Tanh:
    """Tanh activation: f(x) = tanh(x)

    Applies the hyperbolic tangent function element-wise."""
    
    def parameters(self):
        """Return empty list (activations have no learnable parameters)"""
        return []

    def forward(self, X: Tensor) -> Tensor:
        """Forward pass of the Tanh activation function.

        Approach:
        1. use np.tanh for element-wise hyperbolic tangent.
        2. Return the result wrapped in new tensor.

        Example:
        >>> tanh = Tanh()
        >>> x = Tensor(np.array([-1.0, 0.0, 1.0]))
        >>> y = tanh(x)
        >>> print(y)
        Tensor([-0.76159416,  0.0       ,  0.76159416])

        Args:
            X (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after applying Tanh activation.
        """
        result = np.tanh(X.data)
        return Tensor(result)

    def __call__(self, x: Tensor) -> Tensor:
        """Allows the activation to be called like a function."""
        return self.forward(x)

    # def backward(self, grad: Tensor) -> Tensor:
    #     """Backward pass of the Tanh activation function."""
    #     pass

class GELU:
    """GELU activation: f(x) = 0.5x(1 + tanh(sqrt(2/π)(x + 0.0044715x^3)))"""
    
    def parameters(self):
        """Return empty list (activations have no learnable parameters)"""
        return []
    
    def forward(self, X: Tensor) -> Tensor:
        """Forward pass of the GELU activation function.
        
        Approach:
        1. Use approximation : x * sigmoid(1.702 * x).
        2. Compute sigmoid part: 1 / (1 + exp(-1.702 * x)).
        3. Multiply input x with sigmoid result.
        4. Return the result wrapped in new tensor.
        
        """
        sigmoid_part = -1.702 * X.data
        sigmoid_part = np.exp(sigmoid_part)
        sigmoid_part = 1 / (1 + sigmoid_part)  #* Extract data for multiplication
        result = X.data * sigmoid_part
        return Tensor(result)
    
    def __call__(self, x: Tensor) -> Tensor:
        """Allows the activation to be called like a function."""
        return self.forward(x)
    
    # def backward(self, grad: Tensor) -> Tensor:
    #     """Backward pass of the GELU activation function."""
    #     pass
    
class Softmax:
    """Softmax activation: f(x_i) = e^(x_i) / Σ e^(x_j)

    Converts logits to probabilities across classes."""
    
    def parameters(self):
        """Return empty list (activations have no learnable parameters)"""
        return []
    
    def forward(self, X: Tensor, dim:int=-1) -> Tensor:
        """Forward pass of the Softmax activation function.
        Approach:
        1. Subtract max for numerical stability: x - max(x).
        2. Compute exponentials: exp(x - max(x)).
        3. sum along dimension: sum(exp_values).
        3. Normalize: exp_values / sum(exp_values).
        4. Return the result wrapped in new tensor.
        """
        x = X.data
        x_shifted = x - np.max(x, axis=dim, keepdims=True)
        exp_x = np.exp(x_shifted)
        sum_exp_x = np.sum(exp_x, axis=dim, keepdims=True)
        result = exp_x / sum_exp_x
        return Tensor(result)
        
    def __call__(self, x: Tensor) -> Tensor:
        """Allows the activation to be called like a function."""
        return self.forward(x)
    
        
        
        
        
        
        
        
        
        
def unit_test_sigmoid():
    sigmoid = Sigmoid()
    print("🧪 Unit Test: Sigmoid Activation ....")
    
    #* Test basic cases
    x = Tensor([[0.0]])
    results = sigmoid.forward(x)
    assert np.allclose(results.data[0], 0.5), f"Expected 0.5, got {results.data[0]}"
    print("🧪 Basic Test Passed!")
        
    #* test range propery - all outputs should be in (0,1)
    x = Tensor(np.array([-1000.0, -10.0, 0.0, 10.0, 1000.0]))
    results = sigmoid.forward(x)
    assert np.allclose(results.data[0], 0, atol=1e-10), f"Expected ~0, got {results.data[0]}"
    assert np.allclose(results.data[-1], 1, atol=1e-10), f"Expected ~1, got {results.data[-1]}"
    print("🧪 Range Test property passed!.")

    #* Test the extreme values
    x = Tensor(np.array([-1000.0, 1000.0]))
    results = sigmoid.forward(x)
    assert np.allclose(results.data, [0.0, 1.0], atol=1e-6), f"Expected [0.0, 1.0], got {results.data}"
    print("🧪 Extreme values test passed!")
    
    print("✅ Sigmoid function works correctly!")
    print("=" * 50)
    
def test_unit_relu():
    """🔬 Test ReLU implementation."""
    print("🔬 Unit Test: ReLU...")

    relu = ReLU()

    # Test mixed positive/negative values
    x = Tensor([[-2, -1, 0, 1, 2]])
    result = relu.forward(x)
    expected = [0, 0, 0, 1, 2]
    assert np.allclose(result.data, expected), f"ReLU failed, expected {expected}, got {result.data}"
    print("🧪 Mixed values test passed!")
    
    # Test all negative
    x = Tensor([-5, -3, -1])
    result = relu.forward(x)
    assert np.allclose(result.data, [0, 0, 0]), "ReLU should zero all negative values"
    print("🧪 All negative values test passed!")
    
    # Test all positive
    x = Tensor([1, 3, 5])
    result = relu.forward(x)
    assert np.allclose(result.data, [1, 3, 5]), "ReLU should preserve all positive values"
    print("🧪 All positive values test passed!")
    
    # Test sparsity property
    x = Tensor([-1, -2, -3, 1])
    result = relu.forward(x)
    zeros = np.sum(result.data == 0)
    assert zeros == 3, f"ReLU should create sparsity, got {zeros} zeros out of 4"
    print("🧪 Sparsity property test passed!")
    
    print("✅ ReLU works correctly!")
    print("=" * 50)
    
def test_unit_tanh():
    """🔬 Test Tanh implementation."""
    print("🔬 Unit Test: Tanh...")
    
    tanh = Tanh()

    # Test zero
    x = Tensor([0.0])
    result = tanh.forward(x)
    assert np.allclose(result.data, [0.0]), f"tanh(0) should be 0, got {result.data}"
    print("🧪 Zero input test passed!")
    
    # Test range property - all outputs should be in (-1, 1)
    x = Tensor([-10, -1, 0, 1, 10])
    result = tanh.forward(x)
    assert np.all(result.data >= -1) and np.all(result.data <= 1), "All tanh outputs should be in [-1, 1]"
    print("🧪 Range property test passed!")
    
    # Test symmetry: tanh(-x) = -tanh(x)
    x = Tensor([2.0])
    pos_result = tanh.forward(x)
    x_neg = Tensor([-2.0])
    neg_result = tanh.forward(x_neg)
    assert np.allclose(pos_result.data, -neg_result.data), "tanh should be symmetric: tanh(-x) = -tanh(x)"
    print("🧪 Symmetry property test passed!")
    
    # Test extreme values
    x = Tensor([-1000, 1000])
    result = tanh.forward(x)
    assert np.allclose(result.data[0], -1, atol=TOLERANCE), "tanh(-∞) should approach -1"
    assert np.allclose(result.data[1], 1, atol=TOLERANCE), "tanh(+∞) should approach 1"
    print("🧪 Extreme values test passed!")
    
    print("✅ Tanh works correctly!")
    print("=" * 50)
    
def test_unit_gelu():
    """🔬 Test GELU implementation."""
    print("🔬 Unit Test: GELU...")

    gelu = GELU()

    # Test zero (should be approximately 0)
    x = Tensor([0.0])
    result = gelu.forward(x)
    assert np.allclose(result.data, [0.0], atol=TOLERANCE), f"GELU(0) should be ≈0, got {result.data}"
    print("🧪 Zero input test passed!")
    
    # Test positive values (should be roughly preserved)
    x = Tensor([1.0])
    result = gelu.forward(x)
    assert result.data[0] > 0.8 and result.data[0] < 0.8459, f"GELU(1) should be ≈0.84, got {result.data[0]}"
    print("🧪 Positive input test passed!")
    
    # Test negative values (should be small but not zero)
    x = Tensor([-1.0])
    result = gelu.forward(x)
    assert result.data[0] < 0 and result.data[0] > -0.2, f"GELU(-1) should be ≈-0.16, got {result.data[0]}"
    print("🧪 Negative input test passed!")
    
    # Test smoothness property (no sharp corners like ReLU)
    x = Tensor([-0.001, 0.0, 0.001])
    result = gelu.forward(x)
    # Values should be close to each other (smooth)
    diff1 = abs(result.data[1] - result.data[0])
    diff2 = abs(result.data[2] - result.data[1])
    assert diff1 < 0.01 and diff2 < 0.01, "GELU should be smooth around zero"
    print("🧪 Smoothness property test passed!")
    
    print("✅ GELU works correctly!")
    print("=" * 50)
    
def test_unit_softmax():
    """🔬 Test Softmax implementation."""
    print("🔬 Unit Test: Softmax...")

    softmax = Softmax()

    # Test basic probability properties
    x = Tensor([1, 2, 3])
    result = softmax.forward(x)

    # Should sum to 1
    assert np.allclose(np.sum(result.data), 1.0), f"Softmax should sum to 1, got {np.sum(result.data)}"
    print("🧪 Sum to one test passed!")
    
    # All values should be positive
    assert np.all(result.data > 0), "All softmax values should be positive"
    print("🧪 Positivity test passed!")
    
    # All values should be less than 1
    assert np.all(result.data < 1), "All softmax values should be less than 1"

    # Largest input should get largest output
    max_input_idx = np.argmax(x.data)
    max_output_idx = np.argmax(result.data)
    assert max_input_idx == max_output_idx, "Largest input should get largest softmax output"
    print("🧪 Basic probability properties test passed!")
    
    # Test numerical stability with large numbers
    x = Tensor([1000, 1001, 1002])  # Would overflow without max subtraction
    result = softmax.forward(x)
    assert np.allclose(np.sum(result.data), 1.0), "Softmax should handle large numbers"
    assert not np.any(np.isnan(result.data)), "Softmax should not produce NaN"
    assert not np.any(np.isinf(result.data)), "Softmax should not produce infinity"
    print("🧪 Numerical stability test passed!")
    
    # Test with 2D tensor (batch dimension)
    x = Tensor([[1, 2], [3, 4]])
    result = softmax.forward(x, dim=-1)  # Softmax along last dimension
    assert result.shape == (2, 2), "Softmax should preserve input shape"
    # Each row should sum to 1
    row_sums = np.sum(result.data, axis=-1)
    assert np.allclose(row_sums, [1.0, 1.0]), "Each row should sum to 1"
    print("🧪 2D tensor test passed!")
    
    print("✅ Softmax works correctly!")
    print("=" * 50)
    
if __name__ == "__main__":
    print("🧪 Running unit tests...")
    print("=" * 50)
    unit_test_sigmoid()
    test_unit_relu()
    test_unit_gelu()
    test_unit_tanh()
    test_unit_softmax()
    
    
    print("=" * 50)
    print("🧪 All tests completed.")