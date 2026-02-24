#%% 
import numpy as np
import minitorch
from minitorch.activations.activations import ReLU, Sigmoid, Tanh, Softmax
from minitorch.nn.layers import Sequential, Linear, Dropout, Flatten
from minitorch.tensor.tensor import Tensor
from minitorch.dataloaders.dataloader import Dataset, TensorDataset
from minitorch.losses.losses import MSE, BinaryCrossEntropy, SoftMaxCrossEntropy
from minitorch.optimizers.optim import SGD
from typing import Tuple

print("Importing minitorch...")
# print("minitorch version:", minitorch.__version__)
print("Import successful!\n")
#%%


#%%

def main():
    print("\nHello from minitorch!\n")
    x = Tensor(np.array([[1,2,-3,4,-5,7], 
                    [-6,7,8,-9,10,20],
                    [1,2,-3,4,-5,7],
                    [1,2,-3,4,-5,7]], dtype=np.float32), requires_grad=True)

    # batch_size, in_features, out_features = x.shape[0], x.shape[1], x.shape[1]
    # l1 = Linear(in_features,        out_features,       bias=True)
    # l2 = Linear(out_features,       out_features * 2,   bias=True)
    # l3 = Linear(out_features * 2,   out_features,       bias=True)
    # model = Sequential([l1,l2,l3])
    # out = model(x)
    
    # optimizer = SGD(model.parameters())
    # optimizer.step()
    
    
if __name__ == "__main__":
    main()
    
# %%

