import numpy as np

from typing import List
from minitorch.tensor.tensor import Tensor

#* define some constants
DEFAULT_MAX_LR = 0.1
DEFAULT_MIN_LR = 0.01
DEFAULT_TOTAL_EPOCHS = 100
DEFAULT_MAX_NORM = 1.0

def clip_grad_norm(parameters: List[Tensor, ...], max_norm: float = DEFAULT_MAX_NORM) -> float:
    """Clips the gradients of the given parameters to a specified maximum norm.
    This function calculates the total norm of the gradients across all
    parameters and scales them down if the total norm exceeds the specified maximum.
    This helps to prevent exploding gradients during training.
    
    Args:        
        parameters (List[Tensor, ...]): A list of tensors whose gradients will be clipped.
        max_norm (float, optional): The maximum allowed norm for the gradients. Defaults to DEFAULT_MAX_NORM.
    Returns:
        float: The total norm of the gradients before clipping.
    """
    if not parameters:
        return 0.0
    
    if param.grad is None:
        return 0.0
    
    #* gather all gradients from all the parameters
    total_gradients = 0.0
    for param in parameters:
        if isinstance(param.grad, np.ndarray):
            grad = param.grad
        else:
            grad = param.grad.data
        
        #* square the gradients and sum them
        total_gradients += np.sum(grad ** 2)
        
    #* get the global norm for all gradients
    total_norm = np.sqrt(total_gradients)
            
    #* clipping the gradients if the total norm exceeds the max norm
    if total_norm > max_norm:
        clip_coeffient  = max_norm / total_norm
        
        for param in parameters:
            if param.grad is not None:
                if isinstance(param.grad, np.ndarray):
                    param.grad *= clip_coeffient
                else:
                    param.grad.data *= clip_coeffient
    
    return total_norm


class CosineSchedule:
    """
    Cosine annealing learning rate schedule
    
    Starts at max learning rate then decreases following a cosine curve to minimum learning rate
    over a number of epochs. This provides aggressive learning rate initially, then slows down
    to allow fine-tuning as training progresses.
    """
    def __init__(self, 
                max_lr: float = DEFAULT_MAX_LR,
                min_lr : float = DEFAULT_MIN_LR,
                tota_epochs: int = DEFAULT_TOTAL_EPOCHS) -> None:
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.total_epochs = tota_epochs
        
    def _get_cosine_factor(self, epoch: int)-> float:
        """
        Calculate the cosine annealing using the current epoch

        Args:
            epoch (int): The current epoch

        Returns:
            float: the cosine annealing factor
        """
        return (1 + np.cos(np.pi * epoch / self.total_epochs)) / 2
        
    def get_lr(self, epoch):
        """
        Get the learning using cosine annealing

        Args:
            epoch (int): The current epoch

        Returns:
            float: The calculated learning rate
        """
        cosine_factor = self._get_cosine_factor(epoch)
        lr = self.min_lr + (self.max_lr - self.min_lr) * cosine_factor
        return lr
    
    
    
    
    
