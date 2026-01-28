import numpy as np
from minitorch.tensor.tensor import Tensor

def to_chw(image: np.array) -> np.array:
    """
    Convert image to CHW format.
    Accepts (H,W,C) or (H,W) and returns (C,H,W) or (1,H,W)
    """
    if image.ndim ==2: # (H,W)
        return image[np.newaxis, :, :]
    
    if image.ndim !=3:
        raise ValueError("Expected 2D or 3D image array")
    
    h,w,c = image.shape
    if image.shape[-1] <=4:  # likely (H,W,C)
        chw_image = np.transpose(image, (2,0,1))  # (C,H,W)
        return chw_image
    else:
        return image  # assume already (C,H,W)

class ToTensor:
    """
    Convert a numpy array to a Tensor
    """
    def __call__(self, image: np.array):
        return Tensor(image.astype(np.float32))