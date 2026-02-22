#%% 
import numpy as np
import minitorch
from minitorch.activations.activations import ReLU, Sigmoid, Tanh, Softmax
from minitorch.nn.layers import Sequential, Linear, Dropout, Flatten
from minitorch.tensor.tensor import Tensor
from minitorch.dataloaders.dataloader import Dataset, TensorDataset
from minitorch.losses.losses import MSE, BinaryCrossEntropy, SoftMaxCrossEntropy
from typing import Tuple

print("Importing minitorch...")
# print("minitorch version:", minitorch.__version__)
print("Import successful!\n")
#%%

x = Tensor(np.array([[1,2,-3,4,-5,7], 
                    [-6,7,8,-9,10,20],
                    [1,2,-3,4,-5,7],
                    [1,2,-3,4,-5,7]], dtype=np.float32), requires_grad=True)

#%%

def main():
    print("\nHello from minitorch!\n")
    
    sigmoid = Sigmoid()
    softmax = Softmax()
    act  = sigmoid(x)
    act1 = softmax(x)
    # print(act)
    print(act1.sum(axis=-1))
    
if __name__ == "__main__":
    main()
    
# %%

