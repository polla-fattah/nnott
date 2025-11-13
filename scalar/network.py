import numpy as np
from scalar.layer import Layer, DropoutLayer


class Network:
    def __init__(
        self,
        input_size=28 * 28,        # 784 for 28x28 images
        num_classes=10,
        hidden_sizes=(256, 128, 64),    # expanded default network
        learning_rate=0.01,
        activation='relu',
        hidden_activations=None,
        hidden_dropout=0.2,
    ):
        self.layers = []
        self.sequence = []
        self.training = True
        prev_size = input_size

        hidden_activation_list = self._normalize_hidden_activations(hidden_sizes, activation, hidden_activations)
        dropout_values = self._normalize_dropout(hidden_dropout, len(hidden_sizes))

        for idx, (h, act) in enumerate(zip(hidden_sizes, hidden_activation_list)):
            dense = Layer(prev_size, h, activation=act, learning_rate=learning_rate)
            self.layers.append(dense)
            self.sequence.append(dense)
            drop_p = dropout_values[idx]
            if drop_p > 0.0:
                self.sequence.append(DropoutLayer(p=drop_p))
            prev_size = h

        # output layer (linear logits)
        output_layer = Layer(prev_size, num_classes, activation='linear', learning_rate=learning_rate)
        self.layers.append(output_layer)
        self.sequence.append(output_layer)
        self.num_classes = num_classes

    def _normalize_hidden_activations(self, hidden_sizes, default_activation, overrides):
        if overrides is None:
            return [default_activation] * len(hidden_sizes)
        if isinstance(overrides, str):
            overrides = [overrides]
        if len(overrides) == 1 and len(hidden_sizes) > 1:
            overrides = list(overrides) * len(hidden_sizes)
        if len(overrides) != len(hidden_sizes):
            raise ValueError("hidden_activations must match hidden_sizes length.")
        return [act.lower() for act in overrides]

    def _normalize_dropout(self, dropout_spec, length):
        if dropout_spec is None:
            return [0.0] * length
        if isinstance(dropout_spec, (float, int)):
            vals = [float(dropout_spec)] * length
        elif isinstance(dropout_spec, (list, tuple)):
            vals = [float(v) for v in dropout_spec]
        else:
            raise TypeError("hidden_dropout must be float, list, or tuple.")
        if len(vals) == 1 and length > 1:
            vals = vals * length
        if len(vals) != length:
            raise ValueError("hidden_dropout must have one value or match hidden_sizes length.")
        return [min(0.95, max(0.0, v)) for v in vals]

    def forward(self, inputs):
        x = np.asarray(inputs, dtype=np.float32).flatten()
        for module in self.sequence:
            x = module.forward(x)
        return x

    def backward(self, loss_gradient):
        grad = loss_gradient
        for module in reversed(self.sequence):
            grad = module.backward(grad)

    def predict(self, inputs):
        prev_mode = self.training
        self.eval()
        output = self.forward(inputs)
        if prev_mode:
            self.train()
        return int(np.argmax(output))

    def train(self):
        self.training = True
        for module in self.sequence:
            if hasattr(module, "train"):
                module.train()

    def eval(self):
        self.training = False
        for module in self.sequence:
            if hasattr(module, "eval"):
                module.eval()

    def zero_grad(self):
        for layer in self.layers:
            layer.zero_grad()
