import numpy as np

class RNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize weights and biases
        self.Wxh = np.random.randn(input_size, hidden_size) / np.sqrt(input_size)
        self.Whh = np.random.randn(hidden_size, hidden_size) / np.sqrt(hidden_size)
        self.Why = np.random.randn(hidden_size, output_size) / np.sqrt(hidden_size)
        self.bh = np.zeros((1, hidden_size))
        self.by = np.zeros((1, output_size))

    def forward(self, inputs):
        h_t = np.zeros((1, self.hidden_size))  # Initial hidden state
        outputs = []

        for x_t in inputs:
            x_t = x_t.reshape(1, -1)
            h_t = np.tanh(np.dot(x_t, self.Wxh) + np.dot(h_t, self.Whh) + self.bh)
            y_t = np.dot(h_t, self.Why) + self.by
            outputs.append(y_t)

        return outputs, h_t

    def backward(self, inputs, target, learning_rate=0.1):
        dWxh, dWhh, dWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
        dbh, dby = np.zeros_like(self.bh), np.zeros_like(self.by)
        dhnext = np.zeros_like(self.bh)

        outputs, h_t = self.forward(inputs)
        y_t = outputs[-1]  # Get the last output

        delta_y = y_t - target.reshape(1, -1)
        dWhy += np.dot(h_t.T, delta_y)
        dby += delta_y

        delta_h = np.dot(delta_y, self.Why.T) + dhnext
        delta_h_raw = (1 - h_t ** 2) * delta_h
        dWhh += np.dot(h_t.T, delta_h_raw)
        dbh += delta_h_raw
        dWxh += np.dot(inputs[-1].reshape(1, -1).T, delta_h_raw)

        self.Wxh -= learning_rate * dWxh
        self.Whh -= learning_rate * dWhh
        self.Why -= learning_rate * dWhy
        self.bh -= learning_rate * dbh
        self.by -= learning_rate * dby