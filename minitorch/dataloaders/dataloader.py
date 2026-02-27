#-
# DataLoader - Efficient Data pipelines for Machine Learning Training

#* Essential imports for data loading
import sys
import random
import time
import numpy as np

from typing import Iterator, Tuple, List, Optional, Union
from abc import ABC, abstractmethod
from minitorch.tensor.tensor import Tensor

class Dataset(ABC):
    """
    Abstract base class for all datasets
    
    Provides the fundamental interface that all datasets should and must implement:
        - __len__: Returns the number of samples in the dataset.
        - __getitem__: Returns the sample at a given index.
        
    HINT: 
        Abstract methods force subclasses to implement core functionality
    """
    @abstractmethod
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        pass

    @abstractmethod
    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        """
        Return the sample and its label from the dataset at the given index.

        Args:
            index (int): The index of the sample to retrieve.

        Returns:
            Tuple[Tensor, Tensor]: A tuple containing the sample (features)
                                and its corresponding label.
        """
        pass

class TensorDataset(Dataset):
    """Dataset wrapping tensors for supervised learning.
    
    Each sample(a tuple of tensors) will be retrieved by indexing tensors along the first dimension.
    All tensors must have the same size in their first dimension
    """
    def __init__(self, *tensors: Tensor) -> None:
        super().__init__()
        assert len(tensors) > 0
        self.tensors = tensors
        
        #* validate all tensors have same first dimension
        first_dim = len(tensors[0].data)
        for tensor_index, tensor in enumerate(tensors):
            if tensors[0].shape[0] != first_dim:
                raise ValueError(
                    f"All tensors must have same size in first dimension. "
                    f"Tensor 0: {first_dim}, Tensor {tensor_index}: {len(tensor.data)}"
                )
                
    def __len__(self) -> int:
        return len(self.tensors[0].data)
    
    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        if index >= len(self) or index < 0:
            raise IndexError(f"Index {index} out of range for dataset of size {len(self)}")
        return tuple(tensor[index] for tensor in self.tensors[0])
    
    
class DataLoader:
    """
    Data Loader class supporting shuffling and batching of the data
    """
    def __init__(self, dataset: TensorDataset, batch_size: int, shuffle:bool=True):
        assert len(dataset) > 0, "Dataset cannot be empty"
        assert batch_size > 0, "Batch size must be greater than 0"
        
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
    def __len__(self)-> int:
        """
        Get the number of batches in the dataset given the batcch size and the length
        of the dataset
        """
        
        num_batches = (len(self.dataset) + self.batch_size -1) // self.batch_size
        return num_batches
    
    def __iter__(self):
        """
        Return an iterator over the number of batches
        """
        indeces = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(indeces)
        
        for i in range(0, len(self.dataset), self.batch_size):
            batch_indeces = indeces[i:i+self.batch_size]
            batch = [self.dataset[index] for index in batch_indeces]
            yield self._collate_batch(batch)
            
    def _collate_batch(self, batch: List[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, Tensor]:
        """
        Collate a list of samples into a batch.
        """
        num_tensors = len(batch[0])
        batched_tensors = []
        for tensor_index in range(num_tensors):
            batch_list = [sample[tensor_index].data for sample in batch]
            batch_data = np.stack(batch_list, axis=0)
            batch_tensors = Tensor(batch_data)
            batched_tensors.append(batch_tensors)
            
        return tuple(batched_tensors)
    

    
