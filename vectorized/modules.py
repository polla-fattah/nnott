import numpy as np


ACTIVATION_KINDS = {"relu", "leaky_relu", "tanh", "sigmoid", "gelu", "linear"}


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


class Identity(Module):
    def forward(self, x):
        return np.asarray(x, dtype=np.float32)

    def backward(self, grad_output):
        return np.asarray(grad_output, dtype=np.float32)


class Linear(Module):
    def __init__(self, in_features, out_features, activation_hint=None):
        hint = (activation_hint or "").lower()
        if hint in {"relu", "leaky_relu", "gelu"}:
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


class LeakyReLU(Module):
    def __init__(self, negative_slope=0.01):
        self.negative_slope = float(negative_slope)
        self._mask = None
        self.training = True

    def forward(self, x):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]
        self._mask = (x > 0).astype(np.float32)
        return np.where(x > 0, x, self.negative_slope * x)

    def backward(self, grad_output):
        go = np.asarray(grad_output, dtype=np.float32)
        if go.ndim == 1:
            go = go[None, :]
        negative_mask = 1.0 - self._mask
        return go * (self._mask + negative_mask * self.negative_slope)


class Tanh(Module):
    def __init__(self):
        self._output = None
        self.training = True

    def forward(self, x):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]
        self._output = np.tanh(x)
        return self._output

    def backward(self, grad_output):
        go = np.asarray(grad_output, dtype=np.float32)
        if go.ndim == 1:
            go = go[None, :]
        return go * (1.0 - self._output ** 2)


class Sigmoid(Module):
    def __init__(self):
        self._output = None
        self.training = True

    def forward(self, x):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]
        clipped = np.clip(x, -50.0, 50.0)
        self._output = 1.0 / (1.0 + np.exp(-clipped))
        return self._output

    def backward(self, grad_output):
        go = np.asarray(grad_output, dtype=np.float32)
        if go.ndim == 1:
            go = go[None, :]
        return go * self._output * (1.0 - self._output)


class GELU(Module):
    def __init__(self):
        self._input = None
        self.training = True

    def forward(self, x):
        x = np.asarray(x, dtype=np.float32)
        single = False
        if x.ndim == 1:
            x = x[None, :]
            single = True
        self._input = x
        out = self._approx_gelu(x)
        if single:
            return out[0]
        return out

    def backward(self, grad_output):
        go = np.asarray(grad_output, dtype=np.float32)
        single = False
        if go.ndim == 1:
            go = go[None, :]
            single = True
        grad = go * self._approx_grad(self._input)
        if single:
            return grad[0]
        return grad

    @staticmethod
    def _approx_gelu(x):
        coeff = np.sqrt(2.0 / np.pi)
        return 0.5 * x * (1.0 + np.tanh(coeff * (x + 0.044715 * x ** 3)))

    @staticmethod
    def _approx_grad(x):
        coeff = np.sqrt(2.0 / np.pi)
        inner = coeff * (x + 0.044715 * x ** 3)
        tanh_inner = np.tanh(inner)
        sech2 = 1.0 - tanh_inner ** 2
        return 0.5 * (1.0 + tanh_inner + x * sech2 * coeff * (1 + 3 * 0.044715 * x ** 2))


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


class BatchNorm1D(Module):
    def __init__(self, num_features, momentum=0.1, eps=1e-5):
        self.num_features = int(num_features)
        self.momentum = float(momentum)
        self.eps = float(eps)
        self.gamma = np.ones(self.num_features, dtype=np.float32)
        self.beta = np.zeros(self.num_features, dtype=np.float32)
        self.running_mean = np.zeros(self.num_features, dtype=np.float32)
        self.running_var = np.ones(self.num_features, dtype=np.float32)
        self.training = True

        self._x_centered = None
        self._std_inv = None
        self._x_hat = None
        self._ggamma = np.zeros_like(self.gamma)
        self._gbeta = np.zeros_like(self.beta)

    def forward(self, x):
        x = np.asarray(x, dtype=np.float32)
        single = False
        if x.ndim == 1:
            x = x[None, :]
            single = True
        if self.training:
            mean = x.mean(axis=0)
            var = x.var(axis=0)
            self._x_centered = x - mean
            self._std_inv = 1.0 / np.sqrt(var + self.eps)
            self._x_hat = self._x_centered * self._std_inv
            out = self.gamma * self._x_hat + self.beta
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            self._x_centered = None
            self._std_inv = None
            self._x_hat = None
            out = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
            out = self.gamma * out + self.beta
        if single:
            return out[0]
        return out

    def backward(self, grad_output):
        go = np.asarray(grad_output, dtype=np.float32)
        single = False
        if go.ndim == 1:
            go = go[None, :]
            single = True
        if self._x_hat is None:
            return go
        m = go.shape[0]
        self._gbeta += go.sum(axis=0)
        self._ggamma += np.sum(go * self._x_hat, axis=0)
        d_xhat = go * self.gamma
        dvar = np.sum(d_xhat * self._x_centered * -0.5 * (self._std_inv ** 3), axis=0)
        dmean = np.sum(-d_xhat * self._std_inv, axis=0) + dvar * np.mean(-2.0 * self._x_centered, axis=0)
        dx = (d_xhat * self._std_inv) + (dvar * 2.0 * self._x_centered / m) + (dmean / m)
        if single:
            return dx[0]
        return dx

    def parameters(self):
        return [(self.gamma, self._ggamma), (self.beta, self._gbeta)]

    def zero_grad(self):
        self._ggamma.fill(0.0)
        self._gbeta.fill(0.0)

    def _state_tensors(self):
        return {
            "gamma": self.gamma,
            "beta": self.beta,
            "running_mean": self.running_mean,
            "running_var": self.running_var,
        }


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

    def zero_grad(self):
        self.mask = None


def activation_from_name(name, **kwargs):
    parsed = (name or "relu").strip().lower()
    if parsed not in ACTIVATION_KINDS:
        raise ValueError(f"Unsupported activation '{name}'. Choose from {sorted(ACTIVATION_KINDS)}")
    if parsed == "relu":
        return ReLU()
    if parsed == "leaky_relu":
        return LeakyReLU(negative_slope=kwargs.get("negative_slope", 0.01))
    if parsed == "tanh":
        return Tanh()
    if parsed == "sigmoid":
        return Sigmoid()
    if parsed == "gelu":
        return GELU()
    return Identity()
