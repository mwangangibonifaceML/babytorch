
import numpy as np
import minitorch
from minitorch.activations.activations import ReLU, Sigmoid, Tanh, Softmax
from minitorch.nn.layers import Sequential, Linear, Dropout, Flatten
from minitorch.tensor.tensor import Tensor
from minitorch.dataloaders.dataloader import DataLoader, TensorDataset
from minitorch.losses.losses import MSE, BinaryCrossEntropy, SoftMaxCrossEntropy
from minitorch.optimizers.optim import SGD
from typing import Tuple

print("Importing minitorch...")
# print("minitorch version:", minitorch.__version__)
print("Import successful!\n")

x = Tensor(np.array([[2.0, 3.0, 4.6,7.0],
                    [4.0,5.0,8.0,10.0],
                    [5.6,7.0, 11.1,1.0],
                    [2.0, 3.0,0.0,-1.0],
                    [4.0,5.0,-2.0, -10.0],
                    [5.6,7.0, 11.9,12.0]]), requires_grad=True)
y = Tensor(np.array([1.0, 2.0, 3.0, 3.0, 4.0,5.0]), requires_grad=True)

ds = TensorDataset(x,y)
dataloader = DataLoader(ds, batch_size=2, shuffle=True)


def main():
    print("\nHello from minitorch!\n")
    
    
    for batch_index, (inputs, targets) in enumerate(dataloader):
        print(f'Batch Number: {batch_index+1}, inputs: {inputs} and targets: {targets}')
    
if __name__ == "__main__":
    main()
    

