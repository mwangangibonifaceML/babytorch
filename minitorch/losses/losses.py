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


def stable_softmax(logits_data):
    """compute softmax probabilities with numerical stability
    
    Subtracts the max value per row before exponentiating to prevent overflow
    
    """
    max_logits = np.max(logits_data, axis=1, keepdims=True)
    exp_logits = np.exp(logits_data - max_logits)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


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
        squared_error = np.mean((predictions.data - targets.data) ** 2)
        error = Tensor(squared_error,
                    requires_grad=predictions.requires_grad,
                    _parents=(predictions,)
                    )
        
        def _backward():
            if predictions.requires_grad:
                N = predictions.data.shape[0]
                mse_grad = (2 / N) * (
                    predictions.data - targets.data
                    )
                predictions._add_grad(mse_grad * error.grad)
                
        error._backward = _backward
        return error
    
class SoftMaxCrossEntropy(Loss):
    def __init__(self) -> None:
        pass
    
    def __call__(self, logits: Tensor, targets: Tensor) -> Tensor:
        return self.forward(logits, targets)
    
    def forward(self, logits: Tensor, targets: Tensor) -> np.any:
        #* calculate log probs
        logits_reshape = logits.reshape(-1, logits.shape[-1])
        log_probs = log_softmax(logits_reshape, dim=-1)
        
        #* get batch_size and make targets ints
        batch_size, num_classes = log_probs.shape[0], log_probs.shape[1]
        target_indices = targets.reshape(-1).data.astype(int)
        
        #* get selected log probs
        selected_log_probs = log_probs.data[np.arange(batch_size), target_indices]
        
        #* calculate loss
        neg_log_probs = -np.mean(selected_log_probs) #* pytorch's NLLoss
        loss = Tensor(neg_log_probs,
                    requires_grad=logits.requires_grad,
                    _parents=(logits,))
        
        def _backward():
            if logits.requires_grad:
                stable_probs = stable_softmax(logits_reshape.data)
                one_hot_enocode = np.zeros((batch_size, num_classes), dtype=np.float32)
                one_hot_enocode[np.arange(batch_size), target_indices] = 1.0
                grad_logits = (stable_probs - one_hot_enocode) / batch_size
                logits_grad = (grad_logits * loss.grad).reshape(logits.shape)
                logits._add_grad(logits_grad)
            
        loss._backward = _backward
        return loss
    
class BCEWithLogits(Loss):
    
    def __call__(self, logits: Tensor, targets: Tensor) -> Tensor:
        return self.forward(logits, targets)
    
    def forward(self, logits: Tensor, targets: Tensor) -> np.any:
        #* apply sigmoid to logits to get probabilities
        #* numerical stable sigmoid
        sigmoid_logits = 1 / (1 + np.exp(-np.clip(logits.data, -500, 500)))
        
        #* clip the probabilities to avoid log(0)
        probs_clipped = np.clip(sigmoid_logits, EPSILON, 1 - EPSILON)
        
        #* calculate the binary cross entropy
        #* bce_per_sample = -[y * log(p) + (1-y) * log(1-p)]
        bce_per_sample = -np.sum(targets.data * np.log(probs_clipped) + \
        (1 - targets.data) * np.log(1 - probs_clipped))
        
        loss = Tensor(bce_per_sample, 
                    requires_grad=logits.requires_grad,
                    _parents=(logits,))
        
        def _backward():
            if logits.requires_grad:
                #* gradient w.r.t probabilities
                prob_grad = -(targets.data / probs_clipped) + ((1 - targets.data) / (1 - probs_clipped))
                #* gradient w.r.t logits: prob_grad * sigmoid'(logits)
                sigmoid_deriv = sigmoid_logits * (1 - sigmoid_logits)
                logits_grad = prob_grad * sigmoid_deriv
                logits._add_grad(logits_grad * loss.grad)
            
        loss._backward = _backward
        return loss
    