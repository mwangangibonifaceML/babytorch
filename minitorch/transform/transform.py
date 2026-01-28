import random
import numpy as np
from typing import Tuple, List, Optional
from minitorch.tensor.tensor import Tensor
from minitorch.transform.helpers import to_chw, ToTensor

class RandomState:
    """
    Manages random state for reproducibility across transformations.
    """
    def __init__(self, seed: Optional[int] =None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)   
            self.py_rng = random.Random(seed)
            
        else:
            self.rng = np.random.default_rng()
            self.py_rng = random.Random()
            
class Transform:
    """
    Base class for all transformations.
    """
    def __call__(self, image: np.array) -> np.array:
        raise NotImplementedError("Transform subclasses must implement the __call__ method.")
    
class RandomFlip(Transform):
    """
    Randomly flips images horizontally with given probability. A simple augmentation for 
    most image datasets.
    
    Args:
        prob: Probability of flipping defaults to 0.5
        axis: Axis to flip along, 2 for horizontal, 1 for vertical. Defaults to 2.
        state: Optional RandomState for reproducibility
    """
    def __init__(
        self,
        prob: float =0.5,
        axis: int=2
    ) -> None:
        assert 0.0 <= prob <= 1.0, "Probability must be between 0 and 1"
        self.prob = prob
        self.axis = axis

    def __call__(self, image: np.array) -> np.array:
        return np.flip(image, axis=self.axis).copy()  # Flip along specified axis
        
class RandomCrop(Transform):
    """
    Randomly crops images to a given size.
    
    args:
        crop_size: Tuple of (height, width), pixels to add on each side before cropping
        size: Output size of the crop (height, width)
    """
    def __init__(
        self,
        size: Tuple[int, int],
        state: Optional[RandomState] =None,
        padding: int=0,
        ) -> None:
        if padding <0:
            raise ValueError("Padding must be non-negative")
        if len(size) !=2:
            raise ValueError("Size must be a tuple of (height, width)")
        self.padding = padding
        self.target_h, self.target_w = size
        self.state = state
        
    def __call__(self, image: np.array) -> np.array:
        c,h,w = image.shape
        if self.padding >0:
            padded_image = np.pad(
                image,
                ((0,0), (self.padding, self.padding), (self.padding, self.padding)),
                mode='constant',
                constant_values=0
            )
        else:
            padded_image = image
            
        _, padded_h, padded_w = padded_image.shape
        if padded_h < self.target_h or padded_w < self.target_w:
            raise ValueError("Crop size must be smaller than image size after padding")
        
        max_top = padded_h - self.target_h
        max_left = padded_w - self.target_w

        top = self.state.rng.integers(0, max_top + 1)
        left = self.state.rng.integers(0, max_left + 1)

        cropped_image = padded_image[:, top:top+self.target_h, left:left+self.target_w]

        return cropped_image
    
class Normalization(Transform):
    """
    Normalize a tensor image with mean and standard deviation.
    
    Args:
        mean: Sequence of means for each channel.
        std: Sequence of standard deviations for each channel.
    """
    def __init__(self, mean: List[float], std: List[float]) -> None:
        if len(mean) != len(std):
            raise ValueError("Mean and std must have the same length")
        self.mean = np.array(mean).reshape(-1, 1, 1)
        self.std = np.array(std).reshape(-1, 1, 1)
        
    def __call__(self, image: np.array) -> np.array:
        normalized_image = (image - self.mean) / self.std
        return normalized_image.astype(np.float32)
    

class Compose:
    """
    Composes several transforms together.
    
    Args:
        transforms: List of transform callables to be applied in sequence.
    """
    def __init__(self, transforms: List[callable]) -> None:
        if not all(callable(transform) for transform in transforms):
            raise ValueError("All transforms must be callable")
        self.transforms = transforms
        
    def __call__(self, image: np.array) -> np.array:
        for transform in self.transforms:
            image = transform(image)
        return image