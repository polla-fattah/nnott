import numpy as np

class Neuron:
    def __init__(self, num_inputs, activation='sigmoid', learning_rate=0.01):
        """Initialize neuron with activation-aware weights."""
        act = (activation or 'linear').lower()
        if act in ('relu', 'leaky_relu', 'gelu'):
            std = np.sqrt(2.0 / float(num_inputs))
            self.weights = np.random.randn(num_inputs).astype(np.float32) * std
            self.bias = 0.01 if act == 'relu' else 0.0
        elif act in ('tanh', 'sigmoid'):
            std = np.sqrt(1.0 / float(num_inputs))
            self.weights = np.random.randn(num_inputs).astype(np.float32) * std
            self.bias = 0.0
        else:
            self.weights = np.random.randn(num_inputs).astype(np.float32) * 0.01
            self.bias = 0.0
        self.activation = act
        self.negative_slope = 0.01 if act == 'leaky_relu' else 0.0
        self.learning_rate = learning_rate
        self.last_input = None
        self.last_output = None
        self.last_z = None
        # gradient accumulators for mini-batch updates
        self.grad_w = np.zeros_like(self.weights, dtype=np.float32)
        self.grad_b = 0.0

    def activate(self, x):
        """Apply activation function"""
        if self.activation == 'sigmoid':
            # numerically safer sigmoid
            x = np.clip(x, -50, 50)
            return 1.0 / (1.0 + np.exp(-x))
        elif self.activation == 'relu':
            return np.maximum(0.0, x)
        elif self.activation == 'leaky_relu':
            return np.where(x > 0.0, x, self.negative_slope * x)
        elif self.activation == 'tanh':
            return np.tanh(x)
        elif self.activation == 'gelu':
            coeff = np.sqrt(2.0 / np.pi)
            return 0.5 * x * (1.0 + np.tanh(coeff * (x + 0.044715 * x ** 3)))
        else:
            return x  # linear fallback

    def calculate(self, inputs):
        """Calculate neuron output for given inputs"""
        inputs = np.asarray(inputs, dtype=np.float32)
        if inputs.shape[0] != self.weights.shape[0]:
            raise ValueError("Number of inputs must match number of weights")

        self.last_input = inputs
        total = np.dot(inputs, self.weights) + self.bias
        self.last_z = total
        self.last_output = self.activate(total)
        return self.last_output

    def zero_grad(self):
        self.grad_w.fill(0.0)
        self.grad_b = 0.0
