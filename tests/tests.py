from minitorch.tensor.tensor import Tensor

a = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
b = Tensor([[6, 5, 4], [3, 2, 1]], requires_grad=True)

print("Original Tensor a:")
print(a)
print("\nTransposed Tensor b:")
print(b)

c = a + b
print(c_op)
print("\nResult of a + b:")
print(c)