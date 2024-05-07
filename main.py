# Trung "Jason" Nguyen - ttn190009
# Thien Nguyen - DXN210021
# CS 6375 - Machine Learning - Final Project

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from RNN import RNN

# Load the dataset
url = 'https://drive.google.com/file/d/1wjbHIo3dpP0DLXoThir9qtKj1P2NOKaG/view?usp=sharing'
url = 'https://drive.google.com/uc?id=' + url.split('/')[-2]
dataset = pd.read_csv(url, index_col='Date', parse_dates=['Date'])

# Split data into training and test sets, using data from 2016 and earlier as training data, and data from 2017 and later as test data
traning_dataset = dataset[:'2016'].iloc[:, 1:2].values
test_dataset = dataset['2017':].iloc[:, 1:2].values

# Scaling the training set
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(traning_dataset)

X_train = []
Y_train = []
# Creating a data structure with 60 time-steps and 1 output
for i in range(60, traning_dataset.shape[0]):
    X_train.append(training_set_scaled[i-60:i, 0])
    Y_train.append(training_set_scaled[i, 0])
X_train, Y_train = np.array(X_train), np.array(Y_train)
# Reshaping X_train for efficient modelling
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Train the RNN
rnn = RNN(input_size=1, hidden_size=64, output_size=1)
rnn.train(inputs=X_train, targets=Y_train, num_epochs=50, learn_rate=0.005)


# Process test data
# Use close data of previous 60 days to predict the stock price at time t
original_dataset = pd.concat(
    (dataset["Close"][:'2016'], dataset["Close"]['2017':]), axis=0)
test_dataset_close = original_dataset[len(
    original_dataset)-len(test_dataset) - 60:].values
test_dataset_close = test_dataset_close.reshape(-1, 1)
test_dataset_close = sc.transform(test_dataset_close)

X_test = []
Y_test = []
for i in range(60, test_dataset_close.shape[0]):
    X_test.append(test_dataset_close[i-60:i, 0])
    Y_test.append(test_dataset_close[i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Predicting the stock price with the trained model
predictions = []
for i in range(X_test.shape[0]):
    input_seq = X_test[i]  # Shape: (60, 1)
    output, _ = rnn.forward(input_seq)
    # Assuming output.item() extracts the scalar value
    predictions.append(output.item())

# inverse transform the test set and predictions
Y_test = sc.inverse_transform(np.array(Y_test).reshape(-1, 1))
predictions = sc.inverse_transform(np.array(predictions).reshape(-1, 1))

# Calculate the Mean Squared Error and Accuracy
mse = mean_squared_error(Y_test, predictions)
print("MSE:", mse)


def accuracy(Y_test, Y_pred, threshold=0.05):  # 5% threshold
    accurate_predictions = 0
    for true_value, predicted_value in zip(Y_test, Y_pred):
        if abs(true_value - predicted_value) / true_value <= threshold:
            accurate_predictions += 1
    return accurate_predictions / len(Y_test)


accuracy = accuracy(Y_test, predictions)
print("Accuracy:", accuracy)
