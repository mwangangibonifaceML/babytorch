import math 


class Value:
    def __init__(self, data, label: str=None, _children=(), op:str ='') -> None:
        assert isinstance(data, (float, int)),\
            f'data should be either float or int, you provided {type(data).__name__}'
        self.data = data
        self.label = label
        self._prev = set(_children)
        self.op = op
        self.grad = 0.0
        self._backward = lambda: None
        
    def __repr__(self) -> str:
        return f"Value(data={self.data})"
    
    def __add__(self, other):
        if isinstance(other, (float, int)):
            other = Value(other)
        out = Value(self.data + other.data, _children=(self, other), op='+')
        
        def _backward():
            self.grad = 1.0 * out.grad
            other.grad = 1.0 * out.grad
        out._backward = _backward
        return out
    
    def __mul__(self, other):
        if isinstance(other, (float, int)):
            other = Value(other)
        out = Value(self.data  * other.data, _children=(self, other), op='*')
        
        def _backward():
            self.grad = other.data * out.grad
            other.grad = self.data * out.grad
        out._backward = _backward
        return out
    
    def __pow__(self, pow):
        return Value(self.data ** pow, label='')
        
    def tanh(self):
        num = math.exp(self.data) - math.exp(-self.data)
        den = math.exp(self.data) + math.exp(-self.data)
        out = Value(num/den, label='tanh', _children=(self,), op='tanh')
        
        def _backward():
            self.grad = (1- out.data.__pow__(2)) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()
        def build_topo(root):
            if root not in visited:
                visited.add(root)
                for child in root._prev:
                    build_topo(child)
            topo.append(root)
        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
        