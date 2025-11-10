import numpy as np
from layer import Layer


class Network:
    def __init__(
        self,
        input_size=28 * 28,        # 784 for 28x28 images
        num_classes=10,
        hidden_sizes=(128, 64),    # smaller network for speed/stability
        learning_rate=0.01,
        activation='relu'
    ):
        self.layers = []
        prev_size = input_size

        # hidden layers
        for h in hidden_sizes:
            self.layers.append(
                Layer(prev_size, h, activation=activation, learning_rate=learning_rate)
            )
            prev_size = h

        # output layer
        self.layers.append(
            # keep output layer linear (logits) for CE
            Layer(prev_size, num_classes, activation='linear', learning_rate=learning_rate)
        )
        self.num_classes = num_classes

    def forward(self, inputs):
        # inputs shape (28, 28)
        x = np.asarray(inputs, dtype=np.float32).flatten()  # -> (784,)
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, loss_gradient):
        grad = loss_gradient
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def predict(self, inputs):
        output = self.forward(inputs)
        return int(np.argmax(output))
