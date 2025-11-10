import numpy as np


class Module:
    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad_output):
        raise NotImplementedError

    def parameters(self):
        return []

    def zero_grad(self):
        pass


class Linear(Module):
    def __init__(self, in_features, out_features, activation_hint=None):
        # He init for ReLU-like, otherwise small Gaussian
        if activation_hint == 'relu':
            std = np.sqrt(2.0 / float(in_features))
            W = np.random.randn(in_features, out_features).astype(np.float32) * std
            b = np.full(out_features, 0.01, dtype=np.float32)
        else:
            W = np.random.randn(in_features, out_features).astype(np.float32) * 0.01
            b = np.zeros(out_features, dtype=np.float32)
        self.W = W
        self.b = b
        self._gW = np.zeros_like(self.W, dtype=np.float32)
        self._gb = np.zeros_like(self.b, dtype=np.float32)
        self._x = None

    def forward(self, x):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]
        self._x = x
        return x @ self.W + self.b

    def backward(self, grad_output):
        go = np.asarray(grad_output, dtype=np.float32)
        if go.ndim == 1:
            go = go[None, :]
        # grads
        self._gW += self._x.T @ go
        self._gb += go.sum(axis=0)
        # grad w.r.t input
        return go @ self.W.T

    def parameters(self):
        return [(self.W, self._gW), (self.b, self._gb)]

    def zero_grad(self):
        self._gW.fill(0.0)
        self._gb.fill(0.0)


class ReLU(Module):
    def __init__(self):
        self._mask = None

    def forward(self, x):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]
        self._mask = (x > 0).astype(np.float32)
        return np.maximum(0.0, x)

    def backward(self, grad_output):
        go = np.asarray(grad_output, dtype=np.float32)
        if go.ndim == 1:
            go = go[None, :]
        return go * self._mask


class Sequential(Module):
    def __init__(self, *modules):
        self.modules = list(modules)

    def forward(self, x):
        h = x
        for m in self.modules:
            h = m.forward(h)
        return h

    def backward(self, grad_output):
        g = grad_output
        for m in reversed(self.modules):
            g = m.backward(g)
        return g

    def parameters(self):
        params = []
        for m in self.modules:
            params.extend(m.parameters())
        return params

    def zero_grad(self):
        for m in self.modules:
            m.zero_grad()
