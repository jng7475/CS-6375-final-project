import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from RNN import RNN

# Load the dataset
data = pd.read_csv('GOOGL_2006-01-01_to_2018-01-01.csv')

# Preprocess the data
scaler = MinMaxScaler(feature_range=(0, 1))
data['Close'] = scaler.fit_transform(data['Close'].values.reshape(-1, 1))


# Split the data into training and testing sets
train_size = int(len(data) * 0.8)
train_data = data[:train_size]
test_data = data[train_size:]

# Create input and output sequences
def create_sequences(data, seq_length=30):
    X = []
    y = []
    close_data = data['Close'].values
    data_len = len(close_data)

    for i in range(seq_length, data_len - seq_length):
        X.append(close_data[i - seq_length:i])
        y.append(close_data[i + seq_length])
    return np.array(X), np.array(y).reshape(-1, 1)

seq_length = 30
X_train, y_train = create_sequences(train_data, seq_length)
X_test, y_test = create_sequences(test_data, seq_length)

# Reshape input data
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Define the RNN model
input_size = 1
output_size = 1
hidden_size = 64
rnn = RNN(input_size, output_size, hidden_size)

# Train the RNN
epochs = 100
for epoch in range(epochs):
    for i in range(X_train.shape[0]):
        y_pred, _ = rnn.forward(X_train[i])
        rnn.backprop(y_train[i] - y_pred)

# Prediction
predictions = []
for i in range(X_test.shape[0]):
    y_pred, _ = rnn.forward(X_test[i])
    predictions.append(y_pred[0, 0])

# Inverse transform the predictions
predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

# Evaluate the model
mse = np.mean((y_test.reshape(-1) - predictions.reshape(-1)) ** 2)
print(f"Mean Squared Error: {mse}")