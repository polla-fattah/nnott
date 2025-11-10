import numpy as np

class Neuron:
    def __init__(self, num_inputs, activation='sigmoid', learning_rate=0.01):
        """Initialize neuron with small random weights and zero bias"""
        # small weights help avoid huge activations at the start
        self.weights = (np.random.randn(num_inputs).astype(np.float32)) * 0.01
        self.bias = 0.0
        self.activation = activation
        self.learning_rate = learning_rate
        self.last_input = None
        self.last_output = None
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
        else:
            return x  # linear fallback

    def calculate(self, inputs):
        """Calculate neuron output for given inputs"""
        inputs = np.asarray(inputs, dtype=np.float32)
        if inputs.shape[0] != self.weights.shape[0]:
            raise ValueError("Number of inputs must match number of weights")

        self.last_input = inputs
        total = np.dot(inputs, self.weights) + self.bias
        self.last_output = self.activate(total)
        return self.last_output

    def zero_grad(self):
        self.grad_w.fill(0.0)
        self.grad_b = 0.0
