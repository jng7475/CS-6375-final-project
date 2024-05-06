import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from RNN import RNN

def preprocess_data(filename, seq_length):
    # Load the dataset
    df = pd.read_csv(filename)
    
    # Filter relevant columns
    df = df[['Close']].values
    
    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_scaled = scaler.fit_transform(df)
    
    # Create sequences
    X, y = [], []
    for i in range(len(df_scaled) - seq_length):
        X.append(df_scaled[i:i+seq_length])
        y.append(df_scaled[i+seq_length])
    
    X = np.array(X)
    y = np.array(y)
    
    # Reshape X for RNN input
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    return X, y, scaler


def predict_future(rnn, scaler, initial_data, steps):
    # Reshape initial data
    initial_data = np.reshape(initial_data, (1, len(initial_data), 1))
    
    # Make predictions
    predictions = []
    current_data = initial_data
    
    for i in range(steps):
        output = rnn.forward(current_data)
        predictions.append(output[0][0])
        
        # Prepare input for next step
        current_data = np.roll(current_data, -1)
        current_data[0][-1] = output[0][0]
    
    # Inverse transform the predictions
    predictions = np.array(predictions)
    predictions = np.reshape(predictions, (-1, 1))
    predictions = scaler.inverse_transform(predictions)
    
    return predictions

if __name__ == "__main__":
    # Define sequence length
    seq_length = 50

    # Preprocess data
    X_train, y_train, scaler = preprocess_data('GOOGL_2006-01-01_to_2018-01-01.csv', seq_length)

    # Initialize and train the RNN
    rnn = RNN(input_size=1, hidden_size=50, output_size=1)
    rnn.train(X_train, y_train, epochs=50, batch_size=32)

    # Make predictions
    initial_data = X_train[-1]
    steps = 30
    predicted_prices = predict_future(rnn, scaler, initial_data, steps)

    print(predicted_prices)