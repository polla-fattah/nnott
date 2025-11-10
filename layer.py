import cupy as np
from neuron import Neuron


class Layer:
    def __init__(self, num_inputs, num_neurons, activation='sigmoid', learning_rate=0.01):
        self.neurons = [
            Neuron(num_inputs, activation=activation, learning_rate=learning_rate)
            for _ in range(num_neurons)
        ]
        self.learning_rate = learning_rate
        self.inputs = None
        self.outputs = None

    def forward(self, inputs):
        # make sure inputs are float32
        inputs = np.asarray(inputs, dtype=np.float32)
        self.inputs = inputs
        self.outputs = np.array(
            [neuron.calculate(inputs) for neuron in self.neurons],
            dtype=np.float32
        )
        return self.outputs

    def get_outputs(self):
        return self.outputs

    def backward(self, output_gradient):
        """
        output_gradient: dL/dy_i for each neuron output y_i
        Returns: dL/dx for this layer's input
        """
        output_gradient = np.asarray(output_gradient, dtype=np.float32)
        input_gradient = np.zeros_like(self.inputs, dtype=np.float32)

        for i, neuron in enumerate(self.neurons):
            y = neuron.last_output  # scalar
            # derivative using output
            if neuron.activation == 'sigmoid':
                act_deriv = y * (1.0 - y)
            elif neuron.activation == 'relu':
                act_deriv = 1.0 if y > 0.0 else 0.0
            else:
                act_deriv = 1.0  # linear

            delta = output_gradient[i] * act_deriv  # dL/dz

            old_weights = neuron.weights.copy()

            # update weights & bias (gradient descent)
            neuron.weights -= self.learning_rate * delta * self.inputs
            neuron.bias -= self.learning_rate * delta

            # accumulate gradient wrt inputs
            input_gradient += old_weights * delta

        return input_gradient
