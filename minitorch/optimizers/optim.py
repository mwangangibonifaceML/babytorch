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
    def __init__(self, params: List[Tensor]) -> None:
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
        
    def _extract_grad_data(self)-> List[NDArray]:
        """Extract the gradient of the parameters passed to the optimizer"""
        gradient = []
        for param in self.params:
            gradient.append(param.grad)
        return gradient
    
    def zero_grad(self):
        """Reset the gradients to zero"""
        for parameter in self.params:
            parameter.grad = np.zeros_like(parameter.data)

class SGD(Optimizer):
    def __init__(self,
                params: List[Tensor],
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
        
        if state is None or self.has_momentum():
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
            
            #* extract the gradient for the current parameter
            grad_data = np.array(param.grad.data)
            
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
            param.data = param.data - self.lr * grad_data
            
        #* increament the counter
        self.step_count += 1
        

class Adam(Optimizer):
    def __init__(self,
                params: List[Tensor],
                betas: Tuple[int, int] = (DEFAULT_BETA1, DEFAULT_BETA2),
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
            grad_data = np.array(param.grad.data)
            
            #* perform weight decay if needed
            if self.weight_decay != 0.0:
                grad_data += self.weight_decay * param.data
                
            #* update moments
            m_hat, v_hat = self._update_moments(i, grad_data)
            
            #* update the parameter
            param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            
            
class AdamW(Optimizer):
    def __init__(self,
                params: List[Tensor],
                betas: Tuple[int, int] = (DEFAULT_BETA1, DEFAULT_BETA2),
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
            grad_data = np.array(param.grad.data)
                
            #* update moments
            m_hat, v_hat = self._update_moments(i, grad_data)
            
            #* update the parameter
            param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            param.data *= (1- self.weight_decay)

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
####################### Tests #############################
