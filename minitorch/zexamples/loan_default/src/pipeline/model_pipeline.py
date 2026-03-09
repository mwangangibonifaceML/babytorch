import numpy as np
from minitorch.tensor.tensor import Tensor
from minitorch.nn.layers import Linear, Sequential, Dropout
from minitorch.activations.activations import ReLU, Sigmoid
from minitorch.zexamples.loan_default.src.components.data_transformation import DataTransformation


class LoanDefaultPredictor:
    def __init__(self, in_features: int, out_features: int, drop_out_p: float) -> None:
        self.layers = Sequential(
            [
                #* 1st layer
                Linear(in_features, 128),
                ReLU(),
                Dropout(drop_out_p),
                
                #* 2nd layer
                Linear(128, 64),
                ReLU(),
                
                #* output layer
                Linear(64, out_features),
                Sigmoid()
                
            ]
        )
    
    def forward(self, inputs: Tensor):
        out = self.layers(inputs)
        return out
    
    def __call__(self, inputs: Tensor) -> Tensor:
        return self.forward(inputs)
    
    
    
if __name__ == '__main__':
    data_transformer = DataTransformation()
    train_arr, test_arr = data_transformer.initiate_data_transformation()
    train_tensor, test_tensor = Tensor(train_arr), Tensor(test_arr)

    features, target = train_tensor[:, 1:-2], train_tensor[:, -1]
    print(features, target)
    # loan_model = LoanDefaultPredictor(train_tensor.shape[1], out_features=1, drop_out_p=0.2)
    # out = loan_model(train_tensor)
    # print()
    # print(out)