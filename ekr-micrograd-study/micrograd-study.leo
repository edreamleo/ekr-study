<?xml version="1.0" encoding="utf-8"?>
<!-- Created by Leo: https://leo-editor.github.io/leo-editor/leo_toc.html -->
<leo_file xmlns:leo="https://leo-editor.github.io/leo-editor/namespaces/leo-python-editor/1.1" >
<leo_header file_format="2"/>
<globals/>
<preferences/>
<find_panel_settings/>
<vnodes>
<v t="ekr.20250121054642.1" descendentVnodeUnknownAttributes="7d7100285803000000302e3071017d7102580b0000005f5f626f6f6b6d61726b7371037d7104580700000069735f6475706571054930300a73735805000000302e302e3371067d71075808000000616e6e6f7461746571087d71092858080000007072696f72697479710a4d0f27580a00000070726973657464617465710b580a000000323032312d30332d3330710c75735803000000302e31710d7d710e580b0000005f5f626f6f6b6d61726b73710f7d7110580700000069735f6475706571114930300a7373752e"><vh>Startup &amp; readme</vh>
<v t="ekr.20250121055447.2" descendentVnodeUnknownAttributes="7d71002858010000003071017d7102580b0000005f5f626f6f6b6d61726b7371037d7104580700000069735f6475706571054930300a73735803000000302e3371067d71075808000000616e6e6f7461746571087d71092858080000007072696f72697479710a4d0f27580a00000070726973657464617465710b580a000000323032312d30332d3330710c7573752e"><vh>@settings</vh>
<v t="ekr.20250121055447.4"><vh>@bool check-python-code-on-write = True</vh></v>
<v t="ekr.20250121055447.163"><vh>@data history-list</vh></v>
<v t="ekr.20250121055447.154"><vh>@button backup</vh></v>
<v t="ekr.20250121055447.164" descendentVnodeUnknownAttributes="7d710058010000003071017d71025808000000616e6e6f7461746571037d71042858080000007072696f7269747971054d0f27580a000000707269736574646174657106580a000000323032312d30332d333071077573732e"><vh>@enabled-plugins</vh></v>
</v>
<v t="ekr.20250121055447.161" descendentVnodeUnknownAttributes="7d710058010000003071017d7102580b0000005f5f626f6f6b6d61726b7371037d7104580700000069735f6475706571054930300a7373732e"><vh>Scripts</vh>
<v t="ekr.20250121054955.1"><vh> Recursive import script</vh></v>
</v>
</v>
<v t="ekr.20250121055447.163"></v>
<v t="ekr.20250121055138.1"><vh>study outlines: micrograd</vh>
<v t="ekr.20250121055138.2"><vh>Python3.12..micrograd/engine.py</vh>
<v t="ekr.20250121055138.3"><vh>Value.__init__ &amp; __repr__</vh></v>
<v t="ekr.20250121055138.4"><vh>Value.backward</vh></v>
<v t="ekr.20250121055138.5"><vh>Value.relu</vh></v>
<v t="ekr.20250121055138.6"><vh>Value: fundamental ops</vh>
<v t="ekr.20250121055138.7"><vh>Value.__add__</vh></v>
<v t="ekr.20250121055138.8"><vh>Value.__mul__</vh></v>
<v t="ekr.20250121055138.9"><vh>Value.__pow__</vh></v>
</v>
<v t="ekr.20250121055138.10"><vh>Value: derived ops (wrong op?)</vh>
<v t="ekr.20250121055138.11"><vh>Value.__neg__</vh></v>
<v t="ekr.20250121055138.12"><vh>Value.__radd__</vh></v>
<v t="ekr.20250121055138.13"><vh>Value.__sub__</vh></v>
<v t="ekr.20250121055138.14"><vh>Value.__rsub__</vh></v>
<v t="ekr.20250121055138.15"><vh>Value.__rmul__</vh></v>
<v t="ekr.20250121055138.16"><vh>Value.__truediv__</vh></v>
<v t="ekr.20250121055138.17"><vh>Value.__rtruediv__</vh></v>
<v t="ekr.20250121055138.18"><vh>Value.__repr__</vh></v>
</v>
</v>
<v t="ekr.20250121055138.19"><vh>Python3.12..micrograd/nn.py</vh>
<v t="ekr.20250121055138.20"><vh>class Module</vh>
<v t="ekr.20250121055138.21"><vh>Module.zero_grad</vh></v>
<v t="ekr.20250121055138.22"><vh>Module.parameters</vh></v>
</v>
<v t="ekr.20250121055138.23"><vh>class Neuron</vh>
<v t="ekr.20250121055138.24"><vh>Neuron.__init__ &amp; __repr__</vh></v>
<v t="ekr.20250121055138.25"><vh>Neuron.__call__</vh></v>
<v t="ekr.20250121055138.26"><vh>Neuron.parameters</vh></v>
</v>
<v t="ekr.20250121055138.27"><vh>class Layer</vh>
<v t="ekr.20250121055138.28"><vh>Layer.__init__ &amp; __repr__</vh></v>
<v t="ekr.20250121055138.29"><vh>Layer.__call__</vh></v>
<v t="ekr.20250121055138.30"><vh>Layer.parameters</vh></v>
</v>
<v t="ekr.20250121055138.31"><vh>class MLP</vh>
<v t="ekr.20250121055138.32"><vh>MLP.__init__ &amp; __repr__</vh></v>
<v t="ekr.20250121055138.33"><vh>MLP.__call__</vh></v>
<v t="ekr.20250121055138.34"><vh>MLP.parameters</vh></v>
</v>
</v>
</v>
</vnodes>
<tnodes>
<t tx="ekr.20250121054642.1">@language rest

A study outline for micrograd: 
https://github.com/karpathy/micrograd

@language python
</t>
<t tx="ekr.20250121054955.1">"""Recursively import all python files in a directory and clean the result."""
@language python
@tabwidth -4 # For a better match.
g.cls()
# dir_ = r'C:\Python\Python3.12\Lib\site-packages\micrograd'
dir_ = r'C:\Repos\manim'
c.recursiveImport(
    dir_=dir_,
    kind = '@clean', # '@auto', '@clean', '@nosent','@file',
    recursive = True,
    safe_at_file = True,
    theTypes = ['.py',],
    verbose = True,
)
if 1:
    last = c.lastTopLevel()
    last.expand()
    if last.hasChildren():
        last.firstChild().expand()
    c.redraw(last)
print('Done')</t>
<t tx="ekr.20250121055138.1"></t>
<t tx="ekr.20250121055138.10"></t>
<t tx="ekr.20250121055138.11">def __neg__(self): # -self

    return self * -1
</t>
<t tx="ekr.20250121055138.12">def __radd__(self, other): # other + self
    
    return self + other
</t>
<t tx="ekr.20250121055138.13">def __sub__(self, other): # self - other

    return self + (-other)
</t>
<t tx="ekr.20250121055138.14">def __rsub__(self, other): # other - self
    
    return other + (-self)
</t>
<t tx="ekr.20250121055138.15">def __rmul__(self, other): # other * self

    return self * other
</t>
<t tx="ekr.20250121055138.16">def __truediv__(self, other): # self / other

    return self * other**-1
</t>
<t tx="ekr.20250121055138.17">def __rtruediv__(self, other): # other / self
    
    return other * self**-1
</t>
<t tx="ekr.20250121055138.18"></t>
<t tx="ekr.20250121055138.19">import random
from micrograd.engine import Value

@others
@language python
@tabwidth -4
</t>
<t tx="ekr.20250121055138.2">class Value:
    """ stores a single scalar value and its gradient """
    @others

@language python
@tabwidth -4
</t>
<t tx="ekr.20250121055138.20">class Module:
    @others
</t>
<t tx="ekr.20250121055138.21">def zero_grad(self):
    for p in self.parameters():
        p.grad = 0
</t>
<t tx="ekr.20250121055138.22">def parameters(self):
    return []
</t>
<t tx="ekr.20250121055138.23">class Neuron(Module):
    @others
</t>
<t tx="ekr.20250121055138.24">def __init__(self, nin, nonlin=True):

    self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
    self.b = Value(0)
    self.nonlin = nonlin

def __repr__(self):
    return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"
</t>
<t tx="ekr.20250121055138.25">def __call__(self, x):

    act = sum(
        (wi*xi for wi,xi in zip(self.w, x)),
        self.b
    )
    return act.relu() if self.nonlin else act
</t>
<t tx="ekr.20250121055138.26">def parameters(self):
    return self.w + [self.b]
</t>
<t tx="ekr.20250121055138.27">class Layer(Module):
    @others
</t>
<t tx="ekr.20250121055138.28">def __init__(self, nin, nout, **kwargs):
    self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

def __repr__(self):
    return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"
</t>
<t tx="ekr.20250121055138.29">def __call__(self, x):
    out = [
        n(x) #  Call the Neuron's call method.
        for n in self.neurons
    ]
    return out[0] if len(out) == 1 else out
</t>
<t tx="ekr.20250121055138.3">def __init__(self, data, _children=(), _op=''):

    self.data = data
    self.grad = 0

    # Internal variables used for autograd graph construction...

    self._backward = lambda: None  # EKR: the backward function.
    self._prev = set(_children)
    self._op = _op # the op that produced this node, for graphviz / debugging / etc

def __repr__(self):
    return f"Value(data={self.data}, grad={self.grad})"
</t>
<t tx="ekr.20250121055138.30">def parameters(self):
    return [p for n in self.neurons for p in n.parameters()]
</t>
<t tx="ekr.20250121055138.31">class MLP(Module):
    @others
</t>
<t tx="ekr.20250121055138.32">def __init__(self, nin, nouts):

    sz = [nin] + nouts
    self.layers = [
        Layer(sz[i], sz[i+1], nonlin=i!=len(nouts)-1)
        for i in range(len(nouts))
    ]

def __repr__(self):
    return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
</t>
<t tx="ekr.20250121055138.33">def __call__(self, x):

    for layer in self.layers:
        x = layer(x)  #  Call the Layer's call method.
    return x
</t>
<t tx="ekr.20250121055138.34">def parameters(self):
    return [p for layer in self.layers for p in layer.parameters()]
</t>
<t tx="ekr.20250121055138.4">def backward(self):

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
</t>
<t tx="ekr.20250121055138.5">def relu(self):
    
    

    # out = Value(0 if self.data &lt; 0 else self.data, (self,), 'ReLU')
    data = 0 if self.data &lt; 0 else self.data
    out = Value(data=data, childre=tuple(self), op='ReLU')

    def _backward():
        self.grad += (out.data &gt; 0) * out.grad

    out._backward = _backward
    return out
</t>
<t tx="ekr.20250121055138.6"></t>
<t tx="ekr.20250121055138.7">def __add__(self, other):

    other = other if isinstance(other, Value) else Value(other)
    out = Value(data=self.data + other.data, children=(self, other), op='+')

    def _backward():
        self.grad += out.grad
        other.grad += out.grad

    out._backward = _backward
    return out
</t>
<t tx="ekr.20250121055138.8">def __mul__(self, other):

    other = other if isinstance(other, Value) else Value(other)
    out = Value(data=self.data * other.data, children=(self, other), op='*')

    def _backward():
        self.grad += other.data * out.grad
        other.grad += self.data * out.grad

    out._backward = _backward
    return out
</t>
<t tx="ekr.20250121055138.9">def __pow__(self, other):

    assert isinstance(other, (int, float)), "only supporting int/float powers for now"
    out = Value(data=self.data**other, children=tuple(self), op=f"**{other}")

    def _backward():
        self.grad += (other * self.data**(other-1)) * out.grad

    out._backward = _backward
    return out
</t>
<t tx="ekr.20250121055447.154">"""
Back up this .leo file.

os.environ['LEO_BACKUP'] must be the path to an existing (writable) directory.
"""
c.backup_helper(sub_dir='ekr-study')
</t>
<t tx="ekr.20250121055447.161" __bookmarks="7d7100580700000069735f6475706571014930300a732e"></t>
<t tx="ekr.20250121055447.163">manim

</t>
<t tx="ekr.20250121055447.164" annotate="7d71002858080000007072696f7269747971014d0f27580a000000707269736574646174657102580a000000323032312d30332d33307103752e">@language python

# Recommended plugins, from leoSettings.leo:

plugins_menu.py
# contextmenu.py      # Required by the vim.py and xemacs.py plugins.
mod_scripting.py
nav_qt.py
# Both VR plugins may be enabled.
viewrendered.py
# viewrendered3.py
</t>
<t tx="ekr.20250121055447.2" __bookmarks="7d7100580700000069735f6475706571014930300a732e">@language rest
@wrap

The @settings tree contains all active settings. 

Settings outside this tree have no effect.</t>
<t tx="ekr.20250121055447.4"></t>
</tnodes>
</leo_file>
