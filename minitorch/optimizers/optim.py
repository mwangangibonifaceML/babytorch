############################################################
"""
Optimizers - Sophisticated Learning Algorithms that 
connect gradient calculation in autograd to model training

**Connection map**
Gradients -> Optimizers -> Training.

## Objectives
1. Implement SGD with momentum for stable gradient descent
2. Build Adam Oprimizer with adaptive learning rates
3. Create AdamW optimizer with decoupled weight decay
4. Understand memory and computational trade-offs in optimization algorithms

So, what are optimizers:
    - They are the engines that drive NN learning. The take gradients computed from
        loss functions and use them to update model parameters towards better solutions.
        

"""

import numpy as np
from numpy.typing import NDArray
from typing import List, Optional, Tuple
from minitorch.tensor.tensor import Tensor
from minitorch.nn.layers import Parameter

#* constants for optimizer deafults
DEFAULT_LEARNING_RATE_SGD = 0.01        # Default learning rate for SGD
DEFAULT_LEARNING_RATE_ADAM = 0.001      # Default learning rate for Adam/AdamW
DEFAULT_MOMENTUM = 0.9                  # Default momentum for SGD
DEFAULT_BETA1 = 0.9                     # First moment decay rate for Adam
DEFAULT_BETA2 = 0.999                   # Second moment decay rate for Adam
DEFAULT_EPS = 1e-8                      # Small epsilon for numerical stability in Adam
DEFAULT_WEIGHT_DECAY_ADAMW = 0.01       # Default weight decay for AdamW


class Optimizer:
    """
    Base class for optimizers.
    
    This defines the common interface that all optimizers must implement:
        - zero_grad(): clear gradients from parameter
        - step(): update parameters based on gradients.(The real engine)
    """
    def __init__(self, params: List[Tensor] | List[Parameter]) -> None:
        """
        Initialize the optimizer with the parameters to optimize
        """
        # validate and store the parameters
        if not isinstance(params, list):
            params = list(params)
        self.params = params
        self.step_count = 0
        
    def step(self):
        """
        Update the parameters of the model based on the gradient
        Each optimizer implements its own update rule
        """
        
        raise NotImplementedError(
            f"Abstract method step() not implemented\n"
            f"  ❌ {self.__class__.__name__} inherits from Optimizer but doesn't define step()\n"
            f"  💡 Each optimizer must implement its own update rule (SGD, Adam, etc.)\n"
            f"  🔧 Override step() in your optimizer subclass:\n"
            f"      def step(self):\n"
            f"          for param in self.params:\n"
            f"              if param.grad is not None:\n"
            f"                  param.data -= self.lr * param.grad.data"
        )
        
    def _extract_grad_data(self, gradient: Tensor | NDArray)-> NDArray:
        """Extract the gradient of the parameters passed to the optimizer"""
        if isinstance(gradient, Tensor):
            return gradient.data
        return gradient
    
    def zero_grad(self):
        """Reset the gradients to zero"""
        for parameter in self.params:
            parameter.grad = np.zeros_like(parameter.data)

class SGD(Optimizer):
    def __init__(self,
                params: List[Tensor] | List[Parameter],
                lr: float= DEFAULT_LEARNING_RATE_SGD,
                momentum: float=0.0,
                weight_decay: float= 0.0) -> None:
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.momentum_buffers = [None for _ in self.params]
        
    def has_momentum(self)-> bool:
        """
        Check if this optimizer uses momentum

        Returns:
            bool: True if momentum > 0.0, False otherwise
        """
        if self.momentum > 0.0:
            return True
        return False
    
    def get_momentum_state(self) -> Optional[List | None]:
        """
        Get momentum buffers for checkpointing

        Returns:
            Optional[List]: List of momentum buffers if momentum is enables, 
            None otherwise
        """
        if not self.has_momentum():
            return None
        
        state = [buffer.copy() if buffer is not None else None
                    for buffer in self.momentum_buffers]
        return state
    
    def set_momentum_state(self, state: Optional[List]) -> None:
        """
        Restore momentum buffers for checkpointing

        Args:
            state (Optional[List]): List of momentum buffers or None
        """
        
        if state is None or not self.has_momentum():
            return
        
        if len(state) != len(self.momentum_buffers):
            raise ValueError(
                f"Momentum state length mismatch\n"
                f"  ❌ State has {len(state)} buffers, but optimizer has {len(self.momentum_buffers)} parameters\n"
                f"  💡 Checkpoint was saved with a different model architecture or parameter count\n"
                f"  🔧 Ensure you're loading state into an optimizer with the same number of parameters:\n"
                f"      # Check parameter counts match before restoring\n"
                f"      assert len(saved_state) == len(optimizer.params)"
            )
        for i, buffer in enumerate(state):
            if buffer is not None:
                self.momentum_buffers[i] = buffer.copy()
                
    def step(self):
        """Perform Stochastic Gradient Descent to update the parameters"""    
        #* iterate through all the parameters and update them
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
            
            #* extract the gradient for the current parameter (grad is already ndarray)
            grad_data = self._extract_grad_data(param.grad)
            
            #* apply weight decay for the current parameter
            if self.weight_decay != 0:
                grad_data += self.weight_decay * param.data
            
            #* update momentum buffers
            if self.momentum != 0:
                if self.momentum_buffers[i] is None:
                    self.momentum_buffers[i] = np.zeros_like(param.data)
                    
                #* update momentum: v = momentum * v_prev + grad
                self.momentum_buffers[i] = self.momentum * self.momentum_buffers[i] + grad_data 
                grad_data = self.momentum_buffers[i]
                
            #* update parameter: params = param - lr * grad_data
            param.data -= self.lr * grad_data
            
        #* increament the counter
        self.step_count += 1
        

class Adam(Optimizer):
    """
    Adaptive learning rate optimizer
    
    Adam optimizer fixes SGD's bug, assumption that all paramters need the same learning
    rate for updates. Adam computes individual adaptive learning rates for different
    parameters from estimates of first and second moments of the gradients. This makes
    it effective for problems with sparse gradients or noisy data
    
    """
    def __init__(self,
                params: List[Tensor] | List[Parameter],
                betas: Tuple[float, float] = (DEFAULT_BETA1, DEFAULT_BETA2),
                lr: float= DEFAULT_LEARNING_RATE_ADAM,
                weight_decay: float = 0.0,
                eps: float = DEFAULT_EPS) -> None:
        super().__init__(params)
        self.lr = lr
        self.beta1, self.beat2 = betas[0], betas[1]
        self.weight_decay = weight_decay
        self.eps = eps
        self.m_buffers = [None for _ in self.params]
        self.v_buffers = [None for _ in self.params]
        
    def _update_moments(self, 
                        i: int,
                        gradient_data: np.ndarray
                        )-> Tuple[np.ndarray, ...]:
        """
        Update first and second moment estimates with bias correction.
        
        Computes the exponential moving averages(EMA) of the gradient (first moment)
        and the squared gradient (second moment), then applies bias correction
        to counteract the zero-initialization bias in early training steps.
        """
        #* initialize buffers if its the first time
        if self.m_buffers[i] is None:
            self.m_buffers[i] = np.zeros_like(gradient_data)
            self.v_buffers[i] = np.zeros_like(gradient_data)
            
        #* update biased first and second moment estimate
        self.m_buffers[i] = self.beta1 * self.m_buffers[i] + (1 - self.beta1) * gradient_data
        self.v_buffers[i] = self.beat2 * self.v_buffers[i] + (1 - self.beat2) * (gradient_data ** 2)
        
        #* moments bias correction
        m_bias_correction = 1 - self.beta1 ** self.step_count
        v_bias_correction = 1- self.beat2 ** self.step_count
        
        #* bias corrected moments
        m_hat = self.m_buffers[i] / m_bias_correction
        v_hat = self.v_buffers[i] / v_bias_correction
        
        return m_hat, v_hat
    
    def step(self):
        """
        Update the parameters using adaptive learning rate
        
        Approach:
                Does three things for each parameter
                1) extracts gradient from the parameter.
                2) updates moments for adaptice scaling.
                3) updates the parameters.
        
        """
        #* increment the counter
        self.step_count += 1
        
        #* iterate through all the parameters
        for i, param in enumerate(self.params):
            if param is None:
                continue
            
            #* extract the gradient for the current parameter
            # grad_data = np.array(param.grad.data)
            grad_data = self._extract_grad_data(param.grad)
            
            #* perform weight decay if needed before adaptive scaling
            if self.weight_decay != 0.0:
                grad_data += self.weight_decay * param.data
                
            #* update moments
            m_hat, v_hat = self._update_moments(i, grad_data)
            
            print(m_hat, v_hat)
            #* update the parameter
            #* weight decay get 'adapted' by the learning rate scaling
            param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            
            
class AdamW(Optimizer):
    """
    Adam optimizer with a decoupled weight decay
    
    Fixes a bug in Adam's weight decay implementation by decoupling weight decay
    from the gradient based update, leading to better regularization and is the 
    preffered version in Transformers and most applications.
    
    Key Insight:
        AdamW treats optimization and regularization as separate, independent
        processes, leading to better training dynamics and generalization.
    
    """
    def __init__(self,
                params: List[Tensor] | List[Parameter],
                betas: Tuple[float, float] = (DEFAULT_BETA1, DEFAULT_BETA2),
                lr: float= DEFAULT_LEARNING_RATE_ADAM,
                weight_decay: float = 0.0,
                eps: float = DEFAULT_EPS) -> None:
        """Initialize the AdamW optimizer"""
        super().__init__(params)
        self.lr = lr
        self.beta1, self.beat2 = betas[0], betas[1]
        self.weight_decay = weight_decay
        self.eps = eps
        self.m_buffers = [None for _ in self.params]
        self.v_buffers = [None for _ in self.params]
        
    def _update_moments(self, 
                        i: int,
                        gradient_data: np.ndarray
                        )-> Tuple[np.ndarray, ...]:
        """
        Update first and second moment estimates with bias correction.
        
        Computes the exponential moving averages of the gradient (first moment)
        and the squared gradient (second moment), then applies bias correction
        to counteract the zero-initialization bias in early training steps.
        """
        #* initialize buffers if its the first time
        if self.m_buffers[i] is None:
            self.m_buffers[i] = np.zeros_like(gradient_data)
            self.v_buffers[i] = np.zeros_like(gradient_data)
            
        #* update biased first and second moment estimate
        self.m_buffers[i] = self.beta1 * self.m_buffers[i] + (1 - self.beta1) * gradient_data
        self.v_buffers[i] = self.beat2 * self.v_buffers[i] + (1 - self.beat2) * (gradient_data ** 2)
        
        #* moments bias correction
        m_bias_correction = 1 - self.beta1 ** self.step_count
        v_bias_correction = 1- self.beat2 ** self.step_count
        
        #* bias corrected moments
        m_hat = self.m_buffers[i] / m_bias_correction
        v_hat = self.v_buffers[i] / v_bias_correction
        
        return m_hat, v_hat
    
    def step(self):
        """Update the parameters using adaptive learning rate"""
        #* increment the counter
        self.step_count += 1
        
        #* iterate through all the parameters
        for i, param in enumerate(self.params):
            if param is None:
                continue
            
            #* extract the gradient for the current parameter
            grad_data = self._extract_grad_data(param.grad)
                
            #* update moments
            #* using pure gradients
            m_hat, v_hat = self._update_moments(i, grad_data)
            
            #* update the parameter
            #* weight decay applied after learning rate scaling
            param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            param.data *= (1- self.lr * self.weight_decay)



def test_unit_adam_update_moments():
    """🧪 Test Adam _update_moments computes correct EMA and bias correction."""
    print("🧪 Unit Test: Adam Moment Updates...")
    
    param = Tensor(np.array([1.0,2.0]), requires_grad=True)
    optimizer = Adam([param], lr=0.01, betas=[0.9,0.999], eps=1e-8)
    grad = np.array([2.0,1.0])
    
    #* auto calculation
    optimizer.step_count = 1
    m_hat, v_hat = optimizer._update_moments(0, grad)
    
    #* manual calculation
    m = 0.9 * 0 + (1-0.9) * grad
    v = 0.999 * 0 + (1-0.999) * grad ** 2
    m_bias_corr = 1 - 0.9 ** 1
    v_bias_corr = 1 - 0.999 ** 1
    m_hat_man = m / m_bias_corr
    v_hat_man = v / v_bias_corr
    
    assert np.allclose(m_hat, m_hat_man), f'first moment should equal grad at step 1, got {m_hat}'
    assert np.allclose(v_hat, v_hat_man), f'second moment should equal the square of the grad at step 1, got {v_hat}'
    
    #* step 2
    optimizer.step_count = 2
    m_hat2, v_hat2 = optimizer._update_moments(0, grad)
    
    assert np.allclose(m_hat, m_hat_man), f'first moment should equal grad at step 1, got {m_hat}'
    assert np.allclose(v_hat, v_hat_man), f'second moment should equal the square of the grad at step 1, got {v_hat}'
        
    assert optimizer.m_buffers[0] is not None
    assert optimizer.v_buffers[0] is not None
    print("✅ Adam moment updates work correctly!")
    
    
def test_unit_adam_optimizer():
    """🧪 Test Adam optimizer implementation."""
    print("🧪 Unit Test: Adam Optimizer...")
    
    param = Tensor([5.2,6.5], requires_grad=True)
    optimizer = Adam([param], lr=0.001, betas=[0.9,0.999], eps=1e-8)
    param.grad = np.array([1.0,2.0])
    
    #* auto calculation
    optimizer.step()
    
    #* manual caculation
    #? moments calculation
    m = 0.9 * 0 + (1-0.9) * param.grad
    v = 0.999 * 0 + (1-0.999) * param.grad ** 2
    m_bias_corr = 1 - 0.9 ** 1
    v_bias_corr = 1 - 0.999 ** 1
    m_hat_man = m / m_bias_corr
    v_hat_man = v / v_bias_corr
    
    #* param update
    original_data = param.data.copy()
    expected = original_data - 0.001 * m_hat_man / (np.sqrt(v_hat_man) + 1e-8)
    
    assert np.allclose(param.data, expected, rtol=1e-3), f'Expected auto parameter update to be close to manual update, expected {expected}, got {param}'
    
    print('✅ Adam optimizer works correclty')
    
def test_unit_adamw_optimizer():
    """🧪 Test AdamW optimizer implementation."""
    print("🧪 Unit Test: AdamW Optimizer...")

    # Test AdamW vs Adam difference in weight decay
    # Create identical parameters for comparison
    param_adam = Tensor([1.0, 2.0], requires_grad=True)
    param_adamw = Tensor([1.0, 2.0], requires_grad=True)

    # Create optimizers with same settings
    adam = Adam([param_adam], lr=0.01, weight_decay=0.01)
    adamw = AdamW([param_adamw], lr=0.01, weight_decay=0.01)

    # Set gradients AFTER creating optimizers (optimizer.__init__ resets grad to None)
    param_adam.grad = np.array([0.1, 0.2])
    param_adamw.grad = np.array([0.1, 0.2])

    # Take one step
    adam.step()
    adamw.step()

    # Results should be different due to weight decay implementation
    assert not np.allclose(param_adam.data, param_adamw.data, rtol=1e-6)

    # Test AdamW basic functionality
    param = Tensor([1.0, 2.0], requires_grad=True)
    optimizer = AdamW([param], lr=0.01, weight_decay=0.01)
    
    # Set gradient AFTER creating optimizer
    param.grad = np.array([0.1, 0.2])
    original_data = param.data.copy()

    optimizer.step()

    # Parameter should have changed
    assert not np.array_equal(param.data, original_data)
    assert optimizer.step_count == 1

    # Test that moment buffers are created
    assert optimizer.m_buffers[0] is not None
    assert optimizer.v_buffers[0] is not None

    # Test zero weight decay behaves like Adam
    param1 = Tensor([1.0, 2.0], requires_grad=True)
    param2 = Tensor([1.0, 2.0], requires_grad=True)

    adam_no_wd = Adam([param1], lr=0.01, weight_decay=0.0)
    adamw_no_wd = AdamW([param2], lr=0.01, weight_decay=0.0)

    # Set gradients AFTER creating optimizers
    param1.grad = Tensor([0.1, 0.2])
    param2.grad = Tensor([0.1, 0.2])

    adam_no_wd.step()
    adamw_no_wd.step()

    # Should be very similar (within numerical precision)
    assert np.allclose(param1.data, param2.data, rtol=1e-10)

    print("✅ AdamW optimizer works correctly!")

if __name__ == '__main__':
    test_unit_adam_update_moments()
    test_unit_adam_optimizer()
    test_unit_adamw_optimizer()