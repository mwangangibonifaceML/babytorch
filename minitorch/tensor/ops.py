import numpy as np

class Function:
    def forward(self, *args) -> np.ndarray:
        """Compute forward pass for the operation

        Args:
            One or more numpy array

        Returns:
            np.ndarray: numpy array created by the operation
        """
        raise NotImplementedError()
    
    def __call__(self, *inputs) -> np.ndarray:
        requires_grad = any(input.requires_grad for input in inputs)
        input_data = [input.data for input in inputs]
        output_data = self.forward(*input_data)
        return output_data
        
        
    def backward(self, out_grad, node):
        """Calculates backward pass (gradients)

        Args:
            out_grad upstream gardient flowwing from output to input
            node: Value object holding inputs from forward pass
        """
        pass
        