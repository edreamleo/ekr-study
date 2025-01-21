#@+leo-ver=5-thin
#@+node:ekr.20250121055138.2: * @file engine.py
"""engine.py, adapted from https://github.com/karpathy/micrograd"""

class Value:
    """ stores a single scalar value and its gradient """
    #@+others
    #@+node:ekr.20250121055138.3: ** Value.__init__ & __repr__
    def __init__(self, data, _children=(), _op=''):

        self.data = data
        self.grad = 0

        # Internal variables used for autograd graph construction...

        self._backward = lambda: None  # EKR: the backward function.
        self._prev = set(_children)
        self._op = _op  # the op that produced this node, for graphviz / debugging / etc

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
    #@+node:ekr.20250121055138.4: ** Value.backward
    def backward(self):

        # Topological order all of the children in the graph.
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # For each variable, apply the chain rule to get its gradient.
        self.grad = 1
        for v in reversed(topo):
            v._backward()
    #@+node:ekr.20250121055138.5: ** Value.relu
    def relu(self):



        # out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')
        data = 0 if self.data < 0 else self.data
        out = Value(data=data, childre=tuple(self), op='ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward
        return out
    #@+node:ekr.20250121055138.6: ** Value: fundamental ops
    #@+node:ekr.20250121055138.7: *3* Value.__add__
    def __add__(self, other):

        other = other if isinstance(other, Value) else Value(other)
        out = Value(data=self.data + other.data, children=(self, other), op='+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out
    #@+node:ekr.20250121055138.8: *3* Value.__mul__
    def __mul__(self, other):

        other = other if isinstance(other, Value) else Value(other)
        out = Value(data=self.data * other.data, children=(self, other), op='*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out
    #@+node:ekr.20250121055138.9: *3* Value.__pow__
    def __pow__(self, other):

        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(data=self.data ** other, children=tuple(self), op=f"**{other}")

        def _backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad

        out._backward = _backward
        return out
    #@+node:ekr.20250121055138.10: ** Value: derived ops (wrong op?)
    #@+node:ekr.20250121055138.11: *3* Value.__neg__
    def __neg__(self):  # -self

        return self * -1
    #@+node:ekr.20250121055138.12: *3* Value.__radd__
    def __radd__(self, other):  # other + self

        return self + other
    #@+node:ekr.20250121055138.13: *3* Value.__sub__
    def __sub__(self, other):  # self - other

        return self + (-other)
    #@+node:ekr.20250121055138.14: *3* Value.__rsub__
    def __rsub__(self, other):  # other - self

        return other + (-self)
    #@+node:ekr.20250121055138.15: *3* Value.__rmul__
    def __rmul__(self, other):  # other * self

        return self * other
    #@+node:ekr.20250121055138.16: *3* Value.__truediv__
    def __truediv__(self, other):  # self / other

        return self * other ** -1
    #@+node:ekr.20250121055138.17: *3* Value.__rtruediv__
    def __rtruediv__(self, other):  # other / self

        return other * self ** -1
    #@+node:ekr.20250121055138.18: *3* Value.__repr__
    #@-others

#@@language python
#@@tabwidth -4
#@-leo
