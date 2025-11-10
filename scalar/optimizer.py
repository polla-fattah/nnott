import numpy as np


class Optimizer:
    def step(self, layers, batch_size: int):
        raise NotImplementedError

    def zero_grad(self, layers):
        for layer in layers:
            layer.zero_grad()


class SGD(Optimizer):
    def __init__(self, lr=0.01, momentum=0.0, weight_decay=0.0, nesterov=False):
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self._vel_w = {}
        self._vel_b = {}

    def step(self, layers, batch_size: int):
        for layer in layers:
            for neuron in layer.neurons:
                gid = id(neuron)
                grad_w = neuron.grad_w / float(batch_size)
                grad_b = neuron.grad_b / float(batch_size)

                if self.weight_decay:
                    grad_w = grad_w + self.weight_decay * neuron.weights

                if self.momentum:
                    vw = self._vel_w.get(gid)
                    vb = self._vel_b.get(gid)
                    if vw is None:
                        vw = np.zeros_like(neuron.weights, dtype=np.float32)
                        vb = 0.0
                    vw = self.momentum * vw + grad_w
                    vb = self.momentum * vb + grad_b
                    if self.nesterov:
                        update_w = self.momentum * vw + grad_w
                        update_b = self.momentum * vb + grad_b
                    else:
                        update_w = vw
                        update_b = vb
                    neuron.weights -= self.lr * update_w
                    neuron.bias -= self.lr * update_b
                    self._vel_w[gid] = vw
                    self._vel_b[gid] = vb
                else:
                    neuron.weights -= self.lr * grad_w
                    neuron.bias -= self.lr * grad_b

        self.zero_grad(layers)


class Adam(Optimizer):
    def __init__(self, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self._m_w = {}
        self._v_w = {}
        self._m_b = {}
        self._v_b = {}
        self._t = {}

    def step(self, layers, batch_size: int):
        b1, b2 = self.betas
        for layer in layers:
            for neuron in layer.neurons:
                gid = id(neuron)
                grad_w = neuron.grad_w / float(batch_size)
                grad_b = neuron.grad_b / float(batch_size)

                if self.weight_decay:
                    grad_w = grad_w + self.weight_decay * neuron.weights

                mw = self._m_w.get(gid)
                vw = self._v_w.get(gid)
                mb = self._m_b.get(gid)
                vb = self._v_b.get(gid)
                t = self._t.get(gid, 0) + 1

                if mw is None:
                    mw = np.zeros_like(neuron.weights, dtype=np.float32)
                    vw = np.zeros_like(neuron.weights, dtype=np.float32)
                    mb = 0.0
                    vb = 0.0

                mw = b1 * mw + (1 - b1) * grad_w
                vw = b2 * vw + (1 - b2) * (grad_w * grad_w)
                mb = b1 * mb + (1 - b1) * grad_b
                vb = b2 * vb + (1 - b2) * (grad_b * grad_b)

                mw_hat = mw / (1 - b1 ** t)
                vw_hat = vw / (1 - b2 ** t)
                mb_hat = mb / (1 - b1 ** t)
                vb_hat = vb / (1 - b2 ** t)

                neuron.weights -= self.lr * mw_hat / (np.sqrt(vw_hat) + self.eps)
                neuron.bias -= self.lr * mb_hat / (np.sqrt(vb_hat) + self.eps)

                self._m_w[gid] = mw
                self._v_w[gid] = vw
                self._m_b[gid] = mb
                self._v_b[gid] = vb
                self._t[gid] = t

        self.zero_grad(layers)
