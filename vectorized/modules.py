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

    # Training/eval mode helpers
    def train(self):
        self.training = True
        return self

    def eval(self):
        self.training = False
        return self

    def _state_tensors(self):
        return {}

    def named_children(self):
        for name, value in self.__dict__.items():
            if isinstance(value, Module):
                yield name, value
            elif isinstance(value, (list, tuple)):
                for idx, item in enumerate(value):
                    if isinstance(item, Module):
                        yield f"{name}.{idx}", item

    def state_dict(self):
        state = {}
        params = self._state_tensors()
        if params:
            state["__params__"] = {k: np.array(v, copy=True) for k, v in params.items()}
        for child_name, child in self.named_children():
            child_state = child.state_dict()
            if child_state:
                state[child_name] = child_state
        return state

    def load_state_dict(self, state):
        params = state.get("__params__", {})
        for key, value in params.items():
            tensor = getattr(self, key, None)
            if tensor is None:
                raise KeyError(f"Parameter '{key}' not found in {self.__class__.__name__}")
            tensor[...] = value
        for child_name, child in self.named_children():
            if child_name in state:
                child.load_state_dict(state[child_name])


class Linear(Module):
    def __init__(self, in_features, out_features, activation_hint=None):
        hint = (activation_hint or "").lower()
        if hint in {"relu", "leaky_relu"}:
            std = np.sqrt(2.0 / float(in_features))
            W = np.random.randn(in_features, out_features).astype(np.float32) * std
            b = np.full(out_features, 0.01 if hint == "relu" else 0.0, dtype=np.float32)
        elif hint in {"tanh", "sigmoid"}:
            std = np.sqrt(1.0 / float(in_features))
            W = np.random.randn(in_features, out_features).astype(np.float32) * std
            b = np.zeros(out_features, dtype=np.float32)
        else:
            W = np.random.randn(in_features, out_features).astype(np.float32) * 0.01
            b = np.zeros(out_features, dtype=np.float32)
        self.W = W
        self.b = b
        self._gW = np.zeros_like(self.W, dtype=np.float32)
        self._gb = np.zeros_like(self.b, dtype=np.float32)
        self._x = None
        self.training = True

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

    def _state_tensors(self):
        return {"W": self.W, "b": self.b}


class ReLU(Module):
    def __init__(self):
        self._mask = None
        self.training = True

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
        self.training = True

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

    def train(self):
        self.training = True
        for m in self.modules:
            if hasattr(m, 'train'):
                m.train()
        return self

    def eval(self):
        self.training = False
        for m in self.modules:
            if hasattr(m, 'eval'):
                m.eval()
        return self


class Dropout(Module):
    def __init__(self, p=0.2):
        assert 0.0 <= p < 1.0
        self.p = float(p)
        self.mask = None
        self.training = True

    def forward(self, x):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]
        if self.training and self.p > 0.0:
            keep_prob = 1.0 - self.p
            self.mask = (np.random.rand(*x.shape) < keep_prob).astype(np.float32)
            return (x * self.mask) / keep_prob
        else:
            self.mask = None
            return x

    def backward(self, grad_output):
        go = np.asarray(grad_output, dtype=np.float32)
        if go.ndim == 1:
            go = go[None, :]
        if self.training and self.p > 0.0 and self.mask is not None:
            keep_prob = 1.0 - self.p
            return (go * self.mask) / keep_prob
        else:
            return go
