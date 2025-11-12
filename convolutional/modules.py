import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


class Module:
    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad_output):
        raise NotImplementedError

    def parameters(self):
        return []

    def zero_grad(self):
        pass

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


def _compute_output_dim(size, kernel, stride, padding, dilation=1):
    return ((size + 2 * padding - dilation * (kernel - 1) - 1) // stride) + 1


def _im2col(x, kernel_size, stride, padding):
    kh, kw = kernel_size
    sh, sw = stride
    ph, pw = padding
    N, C, H, W = x.shape
    H_padded = H + 2 * ph
    W_padded = W + 2 * pw
    x_padded = np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)), mode="constant")
    windows = sliding_window_view(x_padded, (kh, kw), axis=(2, 3))
    windows = windows[:, :, ::sh, ::sw, :, :]
    H_out = windows.shape[2]
    W_out = windows.shape[3]
    cols = np.ascontiguousarray(windows.transpose(0, 2, 3, 1, 4, 5))
    cols = cols.reshape(N * H_out * W_out, C * kh * kw)
    cache = (x.shape, H_out, W_out, kernel_size, stride, padding)
    return cols, cache


def _col2im(cols, cache):
    (N, C, H, W), H_out, W_out, (kh, kw), (sh, sw), (ph, pw) = cache
    cols = cols.reshape(N, H_out, W_out, C, kh, kw).transpose(0, 3, 4, 5, 1, 2)
    H_padded = H + 2 * ph
    W_padded = W + 2 * pw
    dx_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
    for i in range(kh):
        i_max = i + sh * H_out
        for j in range(kw):
            j_max = j + sw * W_out
            dx_padded[:, :, i:i_max:sh, j:j_max:sw] += cols[:, :, i, j]
    if ph == 0 and pw == 0:
        return dx_padded
    return dx_padded[:, :, ph:ph + H, pw:pw + W]


class Sequential(Module):
    def __init__(self, *modules):
        self.modules = list(modules)
        self.training = True

    def forward(self, x):
        out = x
        for m in self.modules:
            out = m.forward(out)
        return out

    def backward(self, grad_output):
        grad = grad_output
        for m in reversed(self.modules):
            grad = m.backward(grad)
        return grad

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
            if hasattr(m, "train"):
                m.train()
        return self

    def eval(self):
        self.training = False
        for m in self.modules:
            if hasattr(m, "eval"):
                m.eval()
        return self


class Conv2D(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        kh, kw = kernel_size
        scale = np.sqrt(2.0 / (in_channels * kh * kw))
        self.W = (np.random.randn(out_channels, in_channels, kh, kw).astype(np.float32) * scale)
        self.b = np.zeros(out_channels, dtype=np.float32)
        self._gW = np.zeros_like(self.W)
        self._gb = np.zeros_like(self.b)
        self._cols_cache = None
        self._x_shape = None
        self.training = True

    def forward(self, x):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 3:
            x = x[None, :]
        cols, cache = _im2col(x, self.kernel_size, self.stride, self.padding)
        W_col = self.W.reshape(self.W.shape[0], -1)
        out = cols @ W_col.T + self.b
        N, _, H, W = x.shape
        H_out = _compute_output_dim(H, self.kernel_size[0], self.stride[0], self.padding[0])
        W_out = _compute_output_dim(W, self.kernel_size[1], self.stride[1], self.padding[1])
        out = out.reshape(N, H_out, W_out, self.W.shape[0]).transpose(0, 3, 1, 2)
        self._cols_cache = (cols, cache)
        self._x_shape = x.shape
        return out

    def backward(self, grad_output):
        g = np.asarray(grad_output, dtype=np.float32)
        if g.ndim == 3:
            g = g[None, :]
        cols, cache = self._cols_cache
        N, _, H_out, W_out = g.shape
        grad_flat = g.transpose(0, 2, 3, 1).reshape(-1, self.W.shape[0])
        self._gb += grad_flat.sum(axis=0)
        cols_grad = grad_flat @ self.W.reshape(self.W.shape[0], -1)
        self._gW += (grad_flat.T @ cols).reshape(self.W.shape)
        dx = _col2im(cols_grad, cache)
        return dx

    def parameters(self):
        return [(self.W, self._gW), (self.b, self._gb)]

    def zero_grad(self):
        self._gW.fill(0.0)
        self._gb.fill(0.0)

    def _state_tensors(self):
        return {"W": self.W, "b": self.b}

    def _state_tensors(self):
        return {"W": self.W, "b": self.b}


class BatchNorm2D(Module):
    def __init__(self, num_features, momentum=0.1, eps=1e-5):
        self.gamma = np.ones(num_features, dtype=np.float32)
        self.beta = np.zeros(num_features, dtype=np.float32)
        self.running_mean = np.zeros(num_features, dtype=np.float32)
        self.running_var = np.ones(num_features, dtype=np.float32)
        self.momentum = momentum
        self.eps = eps
        self._cache = None
        self._ggamma = np.zeros_like(self.gamma)
        self._gbeta = np.zeros_like(self.beta)
        self.training = True

    def forward(self, x):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 3:
            x = x[None, :]
        if self.training:
            axes = (0, 2, 3)
            mean = x.mean(axis=axes, keepdims=True)
            var = x.var(axis=axes, keepdims=True)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.squeeze()
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.squeeze()
            x_hat = (x - mean) / np.sqrt(var + self.eps)
            self._cache = (x, x_hat, mean, var)
        else:
            mean = self.running_mean.reshape(1, -1, 1, 1)
            var = self.running_var.reshape(1, -1, 1, 1)
            x_hat = (x - mean) / np.sqrt(var + self.eps)
        out = self.gamma.reshape(1, -1, 1, 1) * x_hat + self.beta.reshape(1, -1, 1, 1)
        return out

    def backward(self, grad_output):
        if not self.training or self._cache is None:
            return grad_output
        x, x_hat, mean, var = self._cache
        N = x.shape[0] * x.shape[2] * x.shape[3]
        g = grad_output
        self._gbeta += g.sum(axis=(0, 2, 3))
        self._ggamma += (g * x_hat).sum(axis=(0, 2, 3))
        dx_hat = g * self.gamma.reshape(1, -1, 1, 1)
        dvar = (-0.5 * (dx_hat * (x - mean)) * np.power(var + self.eps, -1.5)).sum(axis=(0, 2, 3), keepdims=True)
        dmean = (-dx_hat / np.sqrt(var + self.eps)).sum(axis=(0, 2, 3), keepdims=True) + dvar * (-2.0 * (x - mean)).sum(axis=(0, 2, 3), keepdims=True) / N
        dx = dx_hat / np.sqrt(var + self.eps) + dvar * 2.0 * (x - mean) / N + dmean / N
        return dx

    def parameters(self):
        return [(self.gamma, self._ggamma), (self.beta, self._gbeta)]

    def zero_grad(self):
        self._ggamma.fill(0.0)
        self._gbeta.fill(0.0)

    def _state_tensors(self):
        return {"gamma": self.gamma, "beta": self.beta}

    def _state_tensors(self):
        return {
            "gamma": self.gamma,
            "beta": self.beta,
            "running_mean": self.running_mean,
            "running_var": self.running_var,
        }


class ReLU(Module):
    def __init__(self):
        self._mask = None
        self.training = True

    def forward(self, x):
        x = np.asarray(x, dtype=np.float32)
        self._mask = x > 0
        return np.maximum(0.0, x)

    def backward(self, grad_output):
        return grad_output * self._mask


class LeakyReLU(Module):
    def __init__(self, negative_slope=0.01):
        self.negative_slope = negative_slope
        self._mask = None
        self.training = True

    def forward(self, x):
        x = np.asarray(x, dtype=np.float32)
        self._mask = x > 0
        out = np.where(self._mask, x, self.negative_slope * x)
        return out

    def backward(self, grad_output):
        slope = np.where(self._mask, 1.0, self.negative_slope)
        return grad_output * slope


class MaxPool2D(Module):
    def __init__(self, kernel_size=2, stride=2):
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        self.kernel_size = kernel_size
        self.stride = stride
        self._cache = None
        self.training = True

    def forward(self, x):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 3:
            x = x[None, :]
        cols, cache = _im2col(x, self.kernel_size, self.stride, padding=(0, 0))
        kh, kw = self.kernel_size
        C = x.shape[1]
        cols = cols.reshape(-1, C, kh * kw)
        max_vals = cols.max(axis=2)
        mask = (cols == max_vals[:, :, None]).astype(np.float32)
        N = x.shape[0]
        H_out = _compute_output_dim(x.shape[2], kh, self.stride[0], 0)
        W_out = _compute_output_dim(x.shape[3], kw, self.stride[1], 0)
        out = max_vals.reshape(N, H_out, W_out, C).transpose(0, 3, 1, 2)
        self._cache = (cache, mask, cols.shape)
        return out

    def backward(self, grad_output):
        cache, mask, cols_shape = self._cache
        g = grad_output.transpose(0, 2, 3, 1).reshape(cols_shape[0], cols_shape[1])
        grad_cols = np.zeros(mask.shape, dtype=np.float32)
        grad_cols += mask * g[:, :, None]
        grad_cols = grad_cols.reshape(cols_shape[0], cols_shape[1] * mask.shape[2])
        dx = _col2im(grad_cols, cache)
        return dx


class Dropout(Module):
    def __init__(self, p=0.5):
        assert 0.0 <= p < 1.0
        self.p = float(p)
        self.mask = None
        self.training = True

    def forward(self, x):
        if not self.training or self.p == 0.0:
            self.mask = None
            return x
        keep_prob = 1.0 - self.p
        self.mask = (np.random.rand(*x.shape) < keep_prob).astype(np.float32)
        return x * self.mask / keep_prob

    def backward(self, grad_output):
        if not self.training or self.p == 0.0 or self.mask is None:
            return grad_output
        keep_prob = 1.0 - self.p
        return grad_output * self.mask / keep_prob


class Flatten(Module):
    def __init__(self):
        self._orig_shape = None
        self.training = True

    def forward(self, x):
        x = np.asarray(x, dtype=np.float32)
        self._orig_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, grad_output):
        return grad_output.reshape(self._orig_shape)


class Dense(Module):
    def __init__(self, in_features, out_features, activation_hint=None):
        if activation_hint == "relu":
            std = np.sqrt(2.0 / in_features)
        else:
            std = 0.01
        self.W = np.random.randn(in_features, out_features).astype(np.float32) * std
        self.b = np.zeros(out_features, dtype=np.float32)
        self._x = None
        self._gW = np.zeros_like(self.W)
        self._gb = np.zeros_like(self.b)
        self.training = True

    def forward(self, x):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]
        self._x = x
        return x @ self.W + self.b

    def backward(self, grad_output):
        g = np.asarray(grad_output, dtype=np.float32)
        if g.ndim == 1:
            g = g[None, :]
        self._gW += self._x.T @ g
        self._gb += g.sum(axis=0)
        return g @ self.W.T

    def parameters(self):
        return [(self.W, self._gW), (self.b, self._gb)]

    def zero_grad(self):
        self._gW.fill(0.0)
        self._gb.fill(0.0)


class Softmax(Module):
    def __init__(self, axis=1):
        self.axis = axis
        self._output = None
        self.training = True

    def forward(self, x):
        x = np.asarray(x, dtype=np.float32)
        shifted = x - np.max(x, axis=self.axis, keepdims=True)
        exp = np.exp(shifted)
        probs = exp / np.sum(exp, axis=self.axis, keepdims=True)
        self._output = probs
        return probs

    def backward(self, grad_output):
        # Softmax derivative is complex; return grad_output unchanged for loss layers to handle
        return grad_output


class GlobalAvgPool2D(Module):
    def __init__(self):
        self._input_shape = None
        self.training = True

    def forward(self, x):
        x = np.asarray(x, dtype=np.float32)
        self._input_shape = x.shape
        return x.mean(axis=(2, 3))

    def backward(self, grad_output):
        g = np.asarray(grad_output, dtype=np.float32)
        N, C, H, W = self._input_shape
        g = g[:, :, None, None] / float(H * W)
        return np.broadcast_to(g, self._input_shape)


class DepthwiseConv2D(Module):
    def __init__(self, in_channels, kernel_size, stride=1, padding=0):
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        kh, kw = kernel_size
        scale = np.sqrt(2.0 / (kh * kw))
        self.W = np.random.randn(in_channels, kh, kw).astype(np.float32) * scale
        self._gW = np.zeros_like(self.W)
        self._cache = None
        self.in_channels = in_channels
        self.training = True

    def forward(self, x):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 3:
            x = x[None, :]
        cols, cache = _im2col(x, self.kernel_size, self.stride, self.padding)
        N, _, H, W = x.shape
        H_out = _compute_output_dim(H, self.kernel_size[0], self.stride[0], self.padding[0])
        W_out = _compute_output_dim(W, self.kernel_size[1], self.stride[1], self.padding[1])
        cols = cols.reshape(N * H_out * W_out, self.in_channels, -1)
        out = np.einsum("nck,ck->nc", cols, self.W.reshape(self.in_channels, -1))
        out = out.reshape(N, H_out, W_out, self.in_channels).transpose(0, 3, 1, 2)
        self._cache = (cols, cache, (H_out, W_out))
        return out

    def backward(self, grad_output):
        cols, cache, _ = self._cache
        g = grad_output.transpose(0, 2, 3, 1).reshape(cols.shape[0], cols.shape[1])
        self._gW += np.einsum("nc,nck->ck", g, cols).reshape(self.W.shape)
        grad_cols = g[:, :, None] * self.W.reshape(1, self.in_channels, -1)
        grad_cols = grad_cols.reshape(cols.shape[0], -1)
        dx = _col2im(grad_cols, cache)
        return dx

    def parameters(self):
        return [(self.W, self._gW)]

    def zero_grad(self):
        self._gW.fill(0.0)

    def _state_tensors(self):
        return {"W": self.W}


class LayerNorm2D(Module):
    def __init__(self, num_channels, eps=1e-5):
        self.gamma = np.ones(num_channels, dtype=np.float32)
        self.beta = np.zeros(num_channels, dtype=np.float32)
        self.eps = eps
        self._cache = None
        self._ggamma = np.zeros_like(self.gamma)
        self._gbeta = np.zeros_like(self.beta)
        self.training = True

    def forward(self, x):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 3:
            x = x[None, :]
        x_perm = x.transpose(0, 2, 3, 1)
        mean = x_perm.mean(axis=-1, keepdims=True)
        var = x_perm.var(axis=-1, keepdims=True)
        inv_std = 1.0 / np.sqrt(var + self.eps)
        x_hat = (x_perm - mean) * inv_std
        out = self.gamma.reshape(1, 1, 1, -1) * x_hat + self.beta.reshape(1, 1, 1, -1)
        self._cache = (x_hat, inv_std, x_perm.shape)
        return out.transpose(0, 3, 1, 2)

    def backward(self, grad_output):
        x_hat, inv_std, shape = self._cache
        g_perm = grad_output.transpose(0, 2, 3, 1)
        self._gbeta += g_perm.sum(axis=(0, 1, 2))
        self._ggamma += (g_perm * x_hat).sum(axis=(0, 1, 2))
        grad = g_perm * self.gamma.reshape(1, 1, 1, -1)
        C = shape[-1]
        sum_grad = grad.sum(axis=-1, keepdims=True)
        sum_grad_xhat = (grad * x_hat).sum(axis=-1, keepdims=True)
        dx = (grad - sum_grad / C - x_hat * sum_grad_xhat / C) * inv_std
        return dx.transpose(0, 3, 1, 2)

    def parameters(self):
        return [(self.gamma, self._ggamma), (self.beta, self._gbeta)]

    def zero_grad(self):
        self._ggamma.fill(0.0)
        self._gbeta.fill(0.0)


class SiLU(Module):
    def __init__(self):
        self._sig = None
        self._x = None
        self.training = True

    def forward(self, x):
        x = np.asarray(x, dtype=np.float32)
        self._x = x
        sig = 1.0 / (1.0 + np.exp(-x))
        self._sig = sig
        return x * sig

    def backward(self, grad_output):
        sig = self._sig
        x = self._x
        return grad_output * (sig + x * sig * (1 - sig))


class GELU(Module):
    def __init__(self, approximate=True):
        self.approximate = approximate
        self._x = None
        self.training = True

    def forward(self, x):
        x = np.asarray(x, dtype=np.float32)
        self._x = x
        if self.approximate:
            return 0.5 * x * (1.0 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))
        else:
            return 0.5 * x * (1.0 + erf(x / np.sqrt(2)))

    def backward(self, grad_output):
        x = self._x
        if self.approximate:
            tanh_term = np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3))
            sech2 = 1 - tanh_term ** 2
            term = 0.5 * (1 + tanh_term) + 0.5 * x * sech2 * np.sqrt(2 / np.pi) * (1 + 3 * 0.044715 * x ** 2)
            return grad_output * term
        else:
            raise NotImplementedError("Exact GELU derivative not implemented")


class SqueezeExcite(Module):
    def __init__(self, channels, reduction=4):
        self.channels = channels
        hidden = max(1, channels // reduction)
        self.pool = GlobalAvgPool2D()
        self.fc1 = Dense(channels, hidden)
        self.act = SiLU()
        self.fc2 = Dense(hidden, channels)
        self._sigma = None
        self._input = None
        self.training = True

    def forward(self, x):
        self._input = x
        y = self.pool.forward(x)
        y = self.fc1.forward(y)
        y = self.act.forward(y)
        y = self.fc2.forward(y)
        sigma = 1.0 / (1.0 + np.exp(-y))
        self._sigma = sigma
        scale = sigma.reshape(sigma.shape[0], sigma.shape[1], 1, 1)
        return x * scale

    def backward(self, grad_output):
        sigma = self._sigma
        x = self._input
        grad_input = grad_output * sigma.reshape(sigma.shape[0], sigma.shape[1], 1, 1)
        grad_sigma = (grad_output * x).sum(axis=(2, 3))
        grad_sigma = grad_sigma * sigma * (1 - sigma)
        grad = self.fc2.backward(grad_sigma)
        grad = self.act.backward(grad)
        grad = self.fc1.backward(grad)
        grad = self.pool.backward(grad)
        return grad_input + grad
