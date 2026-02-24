import numpy as np
from typing import Optional

from minitorch.activations.activations import ReLU, Sigmoid, Tanh
from minitorch.tensor.tensor import Tensor
from minitorch.nn.layers import Linear

EPSILON =  1e-7


def log_softmax(tensor: Tensor, dim: Optional[int] = None) -> Tensor:
    """Comptute log-softmax with numerical stability

    Args:
        tensor (Tensor): Input tensor to perform log-softmax
        dim (Optional[int], optional): dimension to perform log-softmax. Defaults to None

    Returns:
        Tensor: log-softmax tensor

    """
    #* find the max value in the tensor
    x = tensor.data
    max_val = np.max(x, axis=dim, keepdims=True)
    
    #* subtract the max to prevent numerical overflow
    #* then calculate the log-softmax
    x_shifted = x - max_val
    exp_x = np.exp(x_shifted)
    exp_x_sum = np.sum(exp_x, axis=dim, keepdims=True)
    
    log_softmax_data = np.log(exp_x_sum + EPSILON)

    #* return log-softmax = tensor - max_val - log_softmax_data
    return Tensor(x - max_val - log_softmax_data)

class Loss:
    def __init__(self) -> None:
        pass
    
    def parameters(self):
        """Return a list of parameters

        Returns:
            list: Empty list since loss don't contribute parameters to the model
        """
        return []
    
    def forward(self, *args, **kwds)-> Tensor:
        raise NotImplementedError(
            'Every class should implement this method independently'
        )
        
    def __call__(self, *args, **kwds) -> Tensor:
        return self.forward(*args, **kwds)

class MSE(Loss):
    def __init__(self) -> None:
        pass

    def __call__(self, predictions: Tensor, targets: Tensor) -> Tensor:
        loss = self.forward(predictions, targets)
        return loss
    
    def forward(self, predictions: Tensor, targets: Tensor) -> np.any:
        diff = predictions.data - targets.data
        error_squared = diff ** 2
        error = Tensor(error_squared.mean(),
                    requires_grad=targets.requires_grad,
                    _parents=(predictions,))
        
        def _backward():
            
            if predictions.requires_grad:
                mse_grad = (2/diff.size) * diff
                predictions._add_grad(mse_grad * error.grad)
                
        error._backward = _backward
        return error
    
class SoftMaxCrossEntropy(Loss):
    def __init__(self) -> None:
        pass
    
    def __call__(self, logits: Tensor, targets: Tensor) -> Tensor:
        np_loss = self.forward(logits, targets)
        return Tensor(np_loss, requires_grad=True)
    
    def forward(self, logits: Tensor, targets: Tensor) -> np.any:
        #* calculate log probs
        log_probs = log_softmax(logits, dim=-1)
        
        #* get batch_size and make targets ints
        batch_size = logits.shape[0]
        target_indices = targets.data.astype(int)
        
        #* get selected log probs
        selected_log_probs = log_probs.data[np.arange(batch_size), target_indices]
        
        #* calculate loss
        neg_log_probs = -np.mean(selected_log_probs)
        loss = Tensor(neg_log_probs,
                    requires_grad=logits.requires_grad,
                    _parents=(logits,))
        
        def _backward():
            if logits.requires_grad:
                grad_logits = (log_probs - selected_log_probs) / batch_size
                logits._add_grad(grad_logits * loss.grad)
            
        loss._backward = _backward
        return loss
    
class BinaryCrossEntropy(Loss):
    
    
    def __call__(self, prediction: Tensor, targets: Tensor) -> Tensor:
        return self.forward(prediction, targets)
    
    def forward(self, prediction: Tensor, targets: Tensor) -> np.any:
        #* clip the predictions to avoid problems with log(0) and log(1)
        prediction_clipped = np.clip(prediction.data, EPSILON, 1 - EPSILON)
        
        #* calculate the binary cross entropy
        #* bce_per_sample = -[y * log(p) + (1-y) * log(1-p)]
        bce_per_sample = -(targets.data * np.log(prediction_clipped) +\
        (1 - targets.data) * np.log(1 - prediction_clipped))
        
        loss = Tensor(bce_per_sample, 
                    requires_grad=prediction.requires_grad,
                    _parents=(prediction))
        
        def _backward():
            if prediction.requires_grad:
                prediction_grad = -(targets.data / prediction_clipped) -\   
                ((1 - targets.data) / (1 - prediction_clipped))
                prediction._add_grad(prediction_grad * loss.grad)
            
        loss._backward = _backward
        return loss
    