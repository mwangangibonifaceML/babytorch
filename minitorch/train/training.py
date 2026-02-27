import numpy as np

#* define some constants
DEFAULT_MAX_LR = 0.1
DEFAULT_MIN_LR = 0.01
DEFAULT_TOTAL_EPOCHS = 100


class CosineSchedule:
    """
    Cosine annealing learning rate schedule
    
    Starts at max learning rate then decreases following a cosine curve to minimum learning rate
    over a number of epochs. This provides aggressive learning rate initially, then fine-tuning
    at the end
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
        return (1+ np.cos(np.pi * epoch / self.total_epochs)) / 2
        
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
    
    
    
    
    
scheduler = CosineSchedule()
lr_1 = scheduler.get_lr(1)
lr_50 = scheduler.get_lr(50)
lr_100 = scheduler.get_lr(100)
print(lr_1, lr_50, lr_100)
