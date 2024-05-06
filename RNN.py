import numpy as np


class RNN:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.001):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Initialize weights
        self.Wxh = np.random.randn(
            hidden_size, input_size) * 0.01  # input to hidden
        self.Whh = np.random.randn(
            hidden_size, hidden_size) * 0.01  # hidden to hidden
        self.Why = np.random.randn(
            output_size, hidden_size) * 0.01  # hidden to output
        self.bh = np.zeros((hidden_size, 1))  # hidden bias
        self.by = np.zeros((output_size, 1))  # output bias

    def forward(self, inputs):
        h = np.zeros((self.hidden_size, 1))
        self.inputs = inputs
        self.hs = {0: h}

        # Forward pass
        for t, x in enumerate(inputs):
            h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh)
            self.hs[t+1] = h

        # Compute output
        output = np.dot(self.Why, h) + self.by
        return output

    def backward(self, doutput):
        dWhy = np.dot(doutput, self.hs[len(self.inputs)].T)
        dby = doutput
        dhnext = np.dot(self.Why.T, doutput)

        dWxh, dWhh, dbh = np.zeros_like(self.Wxh), np.zeros_like(
            self.Whh), np.zeros_like(self.bh)
        dhraw = (1 - self.hs[len(self.inputs)]**2) * dhnext
        for t in reversed(range(len(self.inputs))):
            dWxh += np.dot(dhraw, self.inputs[t].T)
            dWhh += np.dot(dhraw, self.hs[t].T)
            dbh += dhraw
            dhraw = np.dot(self.Whh.T, dhraw) * (1 - self.hs[t]**2)

        return dWxh, dWhh, dWhy, dbh, dby

    def update_parameters(self, dWxh, dWhh, dWhy, dbh, dby):
        # Update parameters using gradient descent
        self.Wxh -= self.learning_rate * dWxh
        self.Whh -= self.learning_rate * dWhh
        self.Why -= self.learning_rate * dWhy
        self.bh -= self.learning_rate * dbh
        self.by -= self.learning_rate * dby

    def train(self, X_train, y_train, epochs=50, batch_size=32):
        for epoch in range(epochs):
            for i in range(0, len(X_train), batch_size):
                X_batch = X_train[i:i+batch_size]
                y_batch = y_train[i:i+batch_size]

                dWxh, dWhh, dWhy, dbh, dby = np.zeros_like(self.Wxh), np.zeros_like(
                    self.Whh), np.zeros_like(self.Why), np.zeros_like(self.bh), np.zeros_like(self.by)

                for j in range(len(X_batch)):
                    inputs = X_batch[j]
                    target = y_batch[j]

                    output = self.forward(inputs)
                    loss = 0.5 * np.sum((output - target) ** 2)
                    doutput = output - target

                    dWxh_temp, dWhh_temp, dWhy_temp, dbh_temp, dby_temp = self.backward(
                        doutput)

                    dWxh += dWxh_temp
                    dWhh += dWhh_temp
                    dWhy += dWhy_temp
                    dbh += dbh_temp
                    dby += dby_temp

                dWxh /= len(X_batch)
                dWhh /= len(X_batch)
                dWhy /= len(X_batch)
                dbh /= len(X_batch)
                dby /= len(X_batch)

                self.update_parameters(dWxh, dWhh, dWhy, dbh, dby)

            if epoch % 10 == 0:
                print(f'Epoch {epoch}, Loss: {loss}')
