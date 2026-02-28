import numpy as np

from typing import List, Callable, Optional, Iterable
from minitorch.tensor.tensor import Tensor
from minitorch.optimizers.optim import Optimizer
from minitorch.losses.losses import Loss
from minitorch.nn.layers import Layer

#* define some constants
DEFAULT_MAX_LR = 0.1
DEFAULT_MIN_LR = 0.01
DEFAULT_TOTAL_EPOCHS = 100
DEFAULT_MAX_NORM = 1.0

def clip_grad_norm(parameters: List[Tensor], max_norm: float = DEFAULT_MAX_NORM) -> float:
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
    
    #* gather all gradients from all the parameters
    total_gradients = 0.0
    for param in parameters:
        if  param.grad is None:
            continue
        
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
    
    return float(total_norm)


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
                total_epochs: int = DEFAULT_TOTAL_EPOCHS) -> None:
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.total_epochs = total_epochs
        
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
    
class Trainer:
    def __init__(self,model, loss_fn, optimizer,scheduler: CosineSchedule | None = None, clip_gradients = True) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.clip_gradients = clip_gradients
        
        #* training state
        self.epoch = 0
        self.step = 0
        self.training_mode = True
        
        #* Histrory tracking
        self.history = {
            'train_loss' : [],
            'eval_loss' : [],
            'learning_rates': []
        }
    
    
    def train_epoch(self, dataloader: Iterable, accumulation_steps: int =1) -> float:
        """
        Train for one epoch

        Args:
            dataloader (Iterable): Iterable yielding input, targets pairs in batches
            accumulation_step (int, optional): Number of batches to accumulate before update. Defaults to 1.
        
        Returns:
            Average loss for the epoch (float)
        """
        ################ STEPS ####################
        #* 1. set model to training = True and training_mode = True.
        #* 2. Loop over batches, calling _process_batch over each.
        #* 3. Every accumulation_steps batches, call _optimizer_update.
        #* 4. Handle remaining accumulated gradients after the loop.
        #* 5. Record average loss, update scheduler, increment epoch.
        
        #* STEP 1
        self.model.training = True
        self.training_mode = True
        
        #* STEP 2
        accumulated_loss = 0.0
        total_loss = 0.0
        num_batches = 0
        
        for i, (inputs, targets) in enumerate(dataloader):
            #* process the batch
            accumulated_loss += self._process_batch(inputs, targets, accumulation_steps)
            
            #* update parameters every accumulation step
            if (i + 1) % accumulation_steps == 0:
                self._optimizer_update()
                total_loss += accumulated_loss
                accumulated_loss = 0.0
                num_batches += 1
                self.step += 1
            
        #* STEP 4
        if accumulated_loss > 0:
            self._optimizer_update()
            total_loss += accumulated_loss
            num_batches += 1
        
        #* STEP 5
        # record the average loss
        average_loss = total_loss/ num_batches
        self.history['train_loss'].append(average_loss)
        
        # update the scheduler
        if self.scheduler is not None:
            learning_rate = self.scheduler.get_lr(self.epoch)
            self.optimizer.lr = learning_rate
            self.history['learning_rates'].append(learning_rate.item())
            
        # increment epochs
        self.epoch += 1
        return average_loss
        
        
    def _process_batch(self, inputs: Tensor, targets: Tensor, accumulation_steps:int)-> float:
        """
        Process a single batch by doing a forward pass, compute loss and backward pass on it.
        
        Args:
            inputs: Input tensor for this batch
            targets: Target tensor for this batch
            accumulation_step: Number of batches per optimizer update (for scaling)

        Returns:
            Scaled loss value (float) for accumulation tracking
        """
        #* forward pass
        predictions = self.model(inputs)
        loss = self.loss_fn(predictions, targets)
        
        #* compute scaled loss for accumulation
        scaled_loss = loss.data / accumulation_steps
        
        #* backward pass
        loss.backward()
        
        return float(scaled_loss)

    def _optimizer_update(self):
        """
        Clip gradients if enabled and step the optimizer.
        """
        if self.clip_gradients:
            params = self.model.parameters()
            clip_grad_norm(params)
            
        self.optimizer.step()
        self.optimizer.zero_grad()
        
    def _get_scheduler_state(self):
        if self.scheduler is None:
            return None
        return {
            'max_lr': self.scheduler.max_lr,
            'min_lr': self.scheduler.min_lr,
            'total_epochs': self.scheduler.total_epochs
        }
        
    def _set_scheduler_state(self, state):
        """Restore scheduler state from checkpoint."""
        if state is None or self.scheduler is None:
            return
        for key, value in state.items():
            if hasattr(self.scheduler, key):
                setattr(self.scheduler, key, value)

    def _get_optimizer_state(self):
        state = {}
        state{'lr'} = self.optimizer.lr
        
        if hasattr(self.optimizer, 'has_momentum') and self.optimizer.has_momentum():
            momentum_state = self.optimizer.get_momentum_state()
            if momentum_state is not None:
                state{'momentum_state'} = momentum_state
                
        if hasattr(self.optimizer, 'weight_decay'):
            state{'weight_decay'} = self.optimizer.weight_decay
        
        return state
    
    def _set_optimizer_state(self, state):
        """Restore optimizer state from checkpoint."""
        if 'lr' in state:
            self.optimizer.lr = state['lr']
        if 'momentum_buffers' in state:
            if hasattr(self.optimizer, 'has_momentum') and self.optimizer.has_momentum():
                self.optimizer.set_momentum_state(state['momentum_buffers'])