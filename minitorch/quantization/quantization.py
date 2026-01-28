#%%

import time
import warnings
import numpy as np

from typing import Tuple, Dict, List, Optional
from minitorch.tensor.tensor import Tensor
from minitorch.activations.activations import ReLU
from minitorch.layers.layers import Linear, Sequential

#* constants for INT8 quantization
INT8_MIN_VALUE = -128
INT8_MAX_VALUE = 127
INT8_RANGE = 256        # number of possible INT8 values (from -128, 127)
EPSILON = 1e-8          # small value for numerical stability

#* constants for memory calculations
BYTES_PER_FLOAT32 = 4
BYTES_PER_INT8 = 1
MBS_TO_BYTES = 1024 * 1024


if __name__ == "__main__":
    print("✅ Quantization module imports complete")
    
    
    
    
#%%
tensor = Tensor(np.array([[1.0,2.0,3.0], [4.0,5.0,6.0]]))

def quantize_int8(tensor: Tensor) -> Tuple[Tensor, float, int]:
    """Quantize FP32 tensor to INT8 using symmetric quantization
    
    Args: 
        tensor (Tensor): Input FP32 tensor to quantize
        
    Returns:
        Tuple (Tensor, float, int): quantized tensor and the quantization parameters
        scale and zero point
    """
    #* calculate dynamic range
    min_val, max_val = float(np.min(tensor.data)), float(np.max(tensor.data))
    if abs(max_val - min_val) < EPSILON:
        scale, zero_point = 1.0, 0
        quantized_tensor = np.zeros_like(tensor.data, dtype=np.int8)
        return (Tensor(quantized_tensor), scale, zero_point)
        
    #* calculate scale and zero_point for standard quantization
    scale = (max_val - min_val) / (INT8_MAX_VALUE - INT8_MIN_VALUE)
    zero_point = 0

    #* clamp zero_point to a valid INT8 value
    zero_point = int(np.clip(zero_point, INT8_MIN_VALUE, INT8_MAX_VALUE))
    #* apply quantization to the tensor
    quantized_tensor = np.round(
                        (tensor.data - zero_point) / scale).astype(np.int8)
    quantized_tensor = np.clip(quantized_tensor, 
                            INT8_MIN_VALUE, INT8_MAX_VALUE)
    return (Tensor(quantized_tensor), scale, zero_point)



#%%
def test_unit_quantize_int8():
    """Test INT8 quantization implementation"""
    print("🔬 Unit Test: INT8 Quantization...")
    
    # Test basic implemenation
    tensor = Tensor(np.array([[1.0,2.0,3.0], [4.0,5.0,6.0]]))
    quant_tensor, S, Z = quantize_int8(tensor)
    print('Original Tensor:\n ', tensor)
    print('Quantized Tensor:\n ',quant_tensor)
    print(f'Quantization Parameters: s={S}, z={Z}')
    print("🧪 Basic Test Passed!")
    
    #Very quantized values are in INT8 range
    assert np.all(quant_tensor.data >= INT8_MIN_VALUE), f'quantized values should be greater or equal to {INT8_MIN_VALUE}'
    assert np.all(quant_tensor.data <= INT8_MAX_VALUE), f'Quantized value should be less or equal to {INT8_MAX_VALUE}'
    assert isinstance(S, float), 'scale should be a floating point number'
    assert isinstance(Z, int), 'zero point should be a int8'
    print("🧪 Range test Passed!")
    
    # Test edge case: constant tensor
    constant_tensor = Tensor(np.array([[2.0, 2.0], [2.0, 2.0]]))
    q_const, scale_const, zp_const = quantize_int8(constant_tensor)
    assert scale_const == 1.0, 'Scale constant should be 1'
    assert zp_const == 0.0, 'Zero point constant should be 0.0'
    print("🧪 Edge case test Passed!")
    
    # Test dequantization preserves approximate values
    dequantized = (quant_tensor.data.astype(np.float32) - Z) * S
    error = np.mean(np.abs(tensor.data - dequantized))
    assert error < 0.25, f'Quantization error too high: expected error < 0.25 for INT8 got error = {error:.4f}'
    print("🧪 Error tolerance test Passed!")
    print("✅ INT8 quantization works correctly!")

    
if __name__ == '__main__':
    test_unit_quantize_int8()
    
#%%
def dequantize_int8(q_tensor: Tensor, scale: float, zp: int)-> Tensor:
    """Dequantized a tensor for INT8 back to FP32
    
    Args:
        q_tensor (Tensor): Input tensor INT8 to dequantize
        scale (float): quantization parameter scale
        zp (int): quantization parameter zero point
        
    Returns:
        Tensor: the dequantized tensor of FP32
    """
    dequantized = (q_tensor.data.astype(np.float32) - zp) * scale
    return Tensor(dequantized)

#%%
def test_unit_dequantize_int8():
    """Test INT8 Dequantization implementation"""
    print("🔬 Unit Test: INT8 dequantization...")
    tensor = Tensor(np.array([[1.0, 10.5, 45.0], [11.0,20.0,32.0]]))
    quant_tensor, S, Z = quantize_int8(tensor)
    deq_tensor = dequantize_int8(quant_tensor, S, Z)
    threshold = 0.25
    assert np.allclose(deq_tensor.data, tensor.data, threshold), 'the two arrays not equal to each other'
    print("🧪 Simple dequantization passed")
    
    # verify round-trip error is small
    error = np.mean(np.abs(tensor.data - deq_tensor.data))
    assert error < 0.25, f'Round-trip error too high: {error}'
    print('🧪 Round trip error test passed')
    
    # test that the restored tensor is of type FP32
    assert deq_tensor.data.dtype == np.float32, 'Restored should be of dtype= float32'
    print('🧪 Round trip error test passed')
    print("✅ INT8 dequantization works correctly!")
    
    
if __name__ == '__main__':
    test_unit_dequantize_int8()
    
    
#%%
class QuantizedLinear:
    def __init__(self, linear_layer: Linear):
        self.original_layer = linear_layer
        
        #* quantize weights
        self.q_weights, self.w_scale, self.w_zero_point = quantize_int8(linear_layer.weight)
        #* fp32 bias
        if linear_layer.bias is not None:
            self.bias = linear_layer.bias
        else:
            self.bias = None
        
        # Store input quantization parameters (set during calibration)
        self.input_scale = None
        self.input_zero_point = None
        
    def memory_usage(self)-> Dict[str, float]:
        """Calculate memory usage in bytes"""
        #* original bytes usage
        original_weight_bytes = self.original_layer.weight.size * BYTES_PER_FLOAT32
        original_bias_bytes = 0.0
        if self.original_layer.bias is not None:
            original_bias_bytes = self.original_layer.bias.size * BYTES_PER_FLOAT32

        #* quantized bytes usage
        quantized_weight_bytes = self.q_weights.data.size * BYTES_PER_INT8
        quantized_bias_bytes = 0.0
        if self.bias is not None:
            quantized_bias_bytes = self.bias.data.size * BYTES_PER_INT8
            
        #* overhead bytes
        overhead_bytes = BYTES_PER_FLOAT32 * 2
        
        #* total bytes usage
        original_bytes_usage = original_weight_bytes + original_bias_bytes
        quantized_bytes_usage = quantized_weight_bytes + quantized_bias_bytes + overhead_bytes
        
        return {
            'original bytes usage': original_bytes_usage,
            'quantized bytes usage': quantized_bytes_usage,
            'compression rate': original_bytes_usage / quantized_bytes_usage if quantized_bytes_usage > 0.0 else 0.0
        }
        
    def calibrate(self, sample_units: List[Tensor]):
        """Calibrate optimal input quantization parameters using sample data."""
        all_units = []
        for input in sample_units:
            all_units.extend(input.data.flatten())
            
        all_inputs = np.array(all_units)
        max = np.percentile(np.abs(all_inputs), 99.9)
        
        #* calculate scale and zero point 
        if max < EPSILON:
            self.input_scale = 1.0
            self.input_zero_point = 0.0
        else:
            self.input_scale = max  / (INT8_RANGE - 1)
            self.input_zero_point = int(INT8_MIN_VALUE / self.input_scale)
            self.input_zero_point = int(np.clip(self.input_zero_point,
                                            INT8_MIN_VALUE, INT8_MAX_VALUE))
            
    def forward(self, inputs: Tensor)-> Tensor:
        """Forward pass with quantized computation. Return FP32 results"""
        assert self.input_scale is not None, 'Layer must be calibrated first'
        
        #* quantize input
        q_inputs = np.round(inputs.data / self.input_scale).astype(np.int8)
        
        #* INT32 GEMM and rescale to float32
        int32_out = q_inputs.astype(np.int32) @ self.q_weights.data.astype(np.int32).T
        output_scale = self.input_scale * self.w_scale
        output_fp32 = int32_out.astype(np.float32) * output_scale
        
        #* add fp32 bias
        if self.bias is not None:
            output_fp32 += self.bias.data
        
        return Tensor(output_fp32)
    
    def __call__(self, inputs: Tensor)-> Tensor:
        return self.forward(inputs)
    
    def parameters(self)->List[Tensor]:
        """Return the quantized parameters"""
        params = [self.q_weights]
        if self.q_bias is not None:
            params.extend([self.q_bias])
        return params
        
            
#%%
def test_unit_quantized_linear():
    """Test quantized linear implementation"""
    print("🔬 Unit Test: quantized linear implementation...")
    X = Tensor(np.array([1.0,2.0,3.0, 4.0, 5.0]))
    in_features, out_features = 5, 2
    original = Linear(in_features, out_features, bias=True)
    original.weight = Tensor(np.random.randn(out_features, in_features) * 0.5)
    original.bias = Tensor(np.random.randn(out_features) * 0.1)
    original_results = original(X)

    q_linear = QuantizedLinear(original)
    samples = [Tensor(np.random.randn(5).astype(np.float32)) for _ in range(128)]
    q_linear.calibrate(samples)
    quantized_results = q_linear(X)
    error = np.mean(np.abs(
        original_results.data - quantized_results.data) / (np.abs(original_results.data) + 1e-6))
    assert error < 0.25, f'Quantization error too high: {error}'
    print('🧪 Quantized implementation passed')
    
if __name__ == '__main__':
    test_unit_quantized_linear()
#%%
x = Tensor(np.linspace(-1, 1, 10))
original = Linear(x.shape[0], 1, bias=True)
q_linear = QuantizedLinear(original)
q_linear.calibrate([x])
np.mean(np.abs(original(x).data - q_linear(x).data))

#%%

#%%
q_linear.w_scale, q_linear.w_zero_point, q_linear.input_scale, q_linear.input_zero_point