#@+leo-ver=5-thin
#@+node:ekr.20250121055138.19: * @file nn.py
"""nn.py, adapted from https://github.com/karpathy/micrograd"""

import random
from micrograd.engine import Value

#@+others
#@+node:ekr.20250121055138.20: ** class Module
class Module:
    #@+others
    #@+node:ekr.20250121055138.21: *3* Module.zero_grad
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0
    #@+node:ekr.20250121055138.22: *3* Module.parameters
    def parameters(self):
        return []
    #@-others
#@+node:ekr.20250121055138.23: ** class Neuron
class Neuron(Module):
    #@+others
    #@+node:ekr.20250121055138.24: *3* Neuron.__init__ & __repr__
    def __init__(self, nin, nonlin=True):

        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0)
        self.nonlin = nonlin

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"
    #@+node:ekr.20250121055138.25: *3* Neuron.__call__
    def __call__(self, x):

        act = sum(
            (wi * xi for wi, xi in zip(self.w, x)),
            self.b
        )
        return act.relu() if self.nonlin else act
    #@+node:ekr.20250121055138.26: *3* Neuron.parameters
    def parameters(self):
        return self.w + [self.b]
    #@-others
#@+node:ekr.20250121055138.27: ** class Layer
class Layer(Module):
    #@+others
    #@+node:ekr.20250121055138.28: *3* Layer.__init__ & __repr__
    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"
    #@+node:ekr.20250121055138.29: *3* Layer.__call__
    def __call__(self, x):
        out = [
            n(x)  #  Call the Neuron's call method.
            for n in self.neurons
        ]
        return out[0] if len(out) == 1 else out
    #@+node:ekr.20250121055138.30: *3* Layer.parameters
    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]
    #@-others
#@+node:ekr.20250121055138.31: ** class MLP
class MLP(Module):
    #@+others
    #@+node:ekr.20250121055138.32: *3* MLP.__init__ & __repr__
    def __init__(self, nin, nouts):

        sz = [nin] + nouts
        self.layers = [
            Layer(sz[i], sz[i + 1], nonlin=i != len(nouts) - 1)
            for i in range(len(nouts))
        ]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
    #@+node:ekr.20250121055138.33: *3* MLP.__call__
    def __call__(self, x):

        for layer in self.layers:
            x = layer(x)  #  Call the Layer's call method.
        return x
    #@+node:ekr.20250121055138.34: *3* MLP.parameters
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    #@-others
#@-others
#@@language python
#@@tabwidth -4
#@-leo
