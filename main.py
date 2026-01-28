import numpy as np
import minitorch
from minitorch.activations.activations import ReLU, Sigmoid, Tanh, Softmax
from minitorch.layers.layers import Sequential, Linear, Dropout
from minitorch.tensor.tensor import Tensor
from minitorch.dataloaders.dataloader import Dataset, TensorDataset
from typing import Tuple

x = Tensor(np.array([[1,2,3,4,5], [6,7,8,9,10]]))


def main():
    print("Hello from tinytorch!\n")
    

    features = [
        Tensor([[1.0, 2.0],[3.0, 4.0],[5.0, 6.0],[7.0, 8.0]], requires_grad=True),
        Tensor([[9.0, 10.0],[11.0, 12.0],[13.0, 14.0],[15.0, 16.0]], requires_grad=True)
        
    ]
    
    labels = [
        Tensor([1.0, 2.0, 3.0, 4.0]),
        Tensor([5.0, 6.0, 7.0, 8.0])
    ]

    tensors = (features, labels)
    ds = TensorDataset(*tensors)
    print(ds[0])
    print()
    print(ds[1])
    
if __name__ == "__main__":
    main()
