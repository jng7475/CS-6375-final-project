import numpy as np


class RNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size  # size of input layer
        self.hidden_size = hidden_size  # size of hidden layer
        self.output_size = output_size  # size of output layer

        # Initialize weights and biases for different layers
        self.w_input_hidden = np.random.randn(hidden_size, input_size) / 100
        self.w_hidden_hidden = np.random.randn(hidden_size, hidden_size) / 100
        self.w_hidden_output = np.random.randn(output_size, hidden_size) / 100

        self.bias_hidden = np.zeros((hidden_size, 1))
        self.bias_output = np.zeros((output_size, 1))

    def forward(self, inputs):
        # values at hidden layer after put input through activation function
        hidden_layer_values = np.zeros((self.hidden_size, 1))
        # save inputs for backpropagation
        self.inputs = inputs
        # map to save hidden states
        self.hidden_states = {0: hidden_layer_values}
        # Forward pass
        for t, input_t in enumerate(inputs):  # Loop through timesteps
            # put input through tanh activation function
            hidden_layer_values = np.tanh(np.dot(self.w_input_hidden, input_t).reshape(
                -1, 1) + np.dot(self.w_hidden_hidden, hidden_layer_values) + self.bias_hidden)
            # save current hidden state
            self.hidden_states[t + 1] = hidden_layer_values
        # Calculate output: output = weight dot values at hidden layer + bias
        output = np.dot(self.w_hidden_output,
                        hidden_layer_values) + self.bias_output

        return output, hidden_layer_values

    def backward(self, doutput, learn_rate):
        # initialize gradients to zeros
        gradient_input_hidden, gradient_hidden_hidden, gradient_hidden_output = np.zeros_like(
            self.w_input_hidden), np.zeros_like(self.w_hidden_hidden), np.zeros_like(self.w_hidden_output)
        gradient_bias_hidden, gradient_bias_output = np.zeros_like(
            self.bias_hidden), np.zeros_like(self.bias_output)
        gradient_hidden_next = np.zeros_like(self.hidden_states[0])

        for t in reversed(range(len(self.inputs))):
            dy = doutput

            # gradient at output layer
            gradient_hidden_output += np.dot(dy, self.hidden_states[t+1].T)

            # gradient at hidden layer
            gradient_hidden = np.dot(
                self.w_hidden_output.T, dy) + gradient_hidden_next

            # Error signal with activation derivative
            # Include derivative of tanh activation
            error_signal = (1 - self.hidden_states[t+1] ** 2) * gradient_hidden

            gradient_bias_hidden += error_signal
            # gradient at input layer
            gradient_input_hidden += np.dot(error_signal,
                                            self.inputs[t].T.reshape(1, 1))
            # gradient at hidden layer
            gradient_hidden_hidden += np.dot(error_signal,
                                             self.hidden_states[t].T)
            gradient_hidden_next = np.dot(self.w_hidden_hidden.T, error_signal)

        # Gradient Clipping (prevents exploding gradients)
        for gradient in [gradient_input_hidden, gradient_hidden_hidden, gradient_hidden_output, gradient_bias_hidden, gradient_bias_output]:
            np.clip(gradient, -5, 5, out=gradient)

        # Update weights
        self.w_input_hidden -= learn_rate * gradient_input_hidden
        self.w_hidden_hidden -= learn_rate * gradient_hidden_hidden
        self.w_hidden_output -= learn_rate * gradient_hidden_output
        self.bias_hidden -= learn_rate * gradient_bias_hidden
        self.bias_output -= learn_rate * gradient_bias_output

    def train(self, inputs, targets, num_epochs, learn_rate):
        for epoch in range(num_epochs):
            loss = 0
            for i in range(len(inputs)):
                input_sequence, target = inputs[i], targets[i]
                output, _ = self.forward(input_sequence)
                loss += 0.5 * np.sum((output - target) ** 2)
                difference = output - target
                self.backward(difference, learn_rate)
            print(f'Epoch {epoch+1}, Loss: {loss/len(inputs)}')
