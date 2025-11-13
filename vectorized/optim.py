import common.backend as backend

xp = backend.xp


class Optimizer:
    def step(self, params, batch_size: int):
        raise NotImplementedError

    def zero_grad(self, params):
        for p, g in params:
            if g is not None:
                g[...] = 0.0


class SGD(Optimizer):
    def __init__(self, lr=0.01, momentum=0.0, weight_decay=0.0, nesterov=False):
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self._vel = {}  # id(param) -> velocity array

    def step(self, params, batch_size: int):
        for p, g in params:
            if g is None:
                continue
            grad = g / float(batch_size)
            if self.weight_decay:
                grad = grad + self.weight_decay * p
            pid = id(p)
            if self.momentum:
                v = self._vel.get(pid)
                if v is None:
                    v = xp.zeros_like(p)
                v = self.momentum * v + grad
                update = self.momentum * v + grad if self.nesterov else v
                p -= self.lr * update
                self._vel[pid] = v
            else:
                p -= self.lr * grad


class Adam(Optimizer):
    def __init__(self, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self._m = {}
        self._v = {}
        self._t = {}

    def step(self, params, batch_size: int):
        b1, b2 = self.betas
        for p, g in params:
            if g is None:
                continue
            grad = g / float(batch_size)
            if self.weight_decay:
                grad = grad + self.weight_decay * p
            pid = id(p)
            m = self._m.get(pid)
            v = self._v.get(pid)
            t = self._t.get(pid, 0) + 1
            if m is None:
                m = xp.zeros_like(p)
                v = xp.zeros_like(p)
            m = b1 * m + (1 - b1) * grad
            v = b2 * v + (1 - b2) * (grad * grad)
            m_hat = m / (1 - b1 ** t)
            v_hat = v / (1 - b2 ** t)
            p -= self.lr * m_hat / (xp.sqrt(v_hat) + self.eps)
            self._m[pid] = m
            self._v[pid] = v
            self._t[pid] = t
