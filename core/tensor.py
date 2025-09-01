# core/tensor.py
# The core of Titan-X: Autograd Engine

class Tensor:
    def __init__(self, data):
        self.data = data
        self.grad = 0
        self._backward = lambda: None
        self._prev = set()

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data)
        
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        out._prev = {self, other}  # ✅ Track creators
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data)  # ✅ Fixed: * not +
        
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        out._prev = {self, other}  # ✅ Track creators
        return out

    def backward(self):
        self.grad = 1.0
        # In real version, we'll add topological sort
        # For now, simple case — assume order is correct
        # Later: build topo sort to call _backward in right order
