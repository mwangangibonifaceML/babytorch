import numpy as np
from typing import Optional

from minitorch.activations.activations import ReLU, Sigmoid, Tanh
from minitorch.tensor.tensor import Tensor
from minitorch.layers.layers import Linear

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
    max_val = np.max(tensor.data, axis=dim, keepdims=True)
    
    #* subtract the max to prevent numerical overflow
    #* then calculate the log-softmax
    log_softmax_data = np.log(np.sum(np.exp(tensor.data - max_val), axis=dim, keepdims=True))

    #* return log-softmax = tensor - max_val - log_softmax_data
    return Tensor(tensor.data - max_val - log_softmax_data)

class MSE:
    def __init__(self) -> None:
        pass

    def __call__(self, predictions: Tensor, targets: Tensor) -> Tensor:
        np_mse = self.forward(predictions, targets)
        return Tensor(np_mse, requires_grad=True)
    
    def forward(self, predictions: Tensor, targets: Tensor) -> np.any:
        error_squared = (predictions.data - targets.data) ** 2
        return error_squared.mean()
    
class CrossEntropy:
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
        loss = -np.mean(selected_log_probs)
        return loss
    
class BinaryCrossEntropy:
    def __init__(self) -> None:
        pass 
    
    def __call__(self, prediction: Tensor, targets: Tensor) -> Tensor:
        np_loss = self.forward(prediction, targets)
        return Tensor(np_loss, requires_grad=True)
    
    def forward(self, prediction: Tensor, targets: Tensor) -> np.any:
        #* clip the predictions to avoid problems with log(0) and log(1)
        prediction_clipped = np.clip(prediction.data, EPSILON, 1 - EPSILON)
        
        #* calculate the binary cross entropy
        log_predicted = np.log(prediction_clipped)
        log_one_minus_predicted = np.log(1 - prediction_clipped)
        one_minus_data = 1 - targets.data
    
        one_half = targets.data * log_predicted
        second_half = one_minus_data * log_one_minus_predicted
        bce_per_sample = -(one_half + second_half)
        return np.mean(bce_per_sample)