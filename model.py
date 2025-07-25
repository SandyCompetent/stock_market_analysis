import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam


def prepare_data_for_lstm(data, feature_columns, target_column, sequence_length, test_size):
    """A unified function to prepare data for LSTM models."""
    if target_column not in feature_columns:
        raise ValueError("The target_column must be in the feature_columns list.")

    target_idx = feature_columns.index(target_column)
    features = data[feature_columns].values

    split_index = int(len(features) * (1 - test_size))
    train_data, test_data = features[:split_index], features[split_index:]

    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)

    X_train, y_train = [], []
    for i in range(sequence_length, len(train_scaled)):
        X_train.append(train_scaled[i - sequence_length:i, :])
        y_train.append(train_scaled[i, target_idx])

    inputs = np.concatenate((train_scaled[-sequence_length:], test_scaled), axis=0)
    X_test, y_test = [], []
    for i in range(sequence_length, len(inputs)):
        X_test.append(inputs[i - sequence_length:i, :])
        y_test.append(inputs[i, target_idx])

    return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test), scaler


def build_single_layer_lstm(input_shape):
    """Builds a single-layer LSTM model."""
    model = Sequential([
        LSTM(units=128, input_shape=input_shape),
        Dropout(0.2),
        Dense(units=1, activation='linear')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model


def build_multi_layer_lstm(input_shape):
    """Builds a stacked, multi-layer LSTM model."""
    model = Sequential([
        LSTM(units=128, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(units=64, return_sequences=False),
        Dropout(0.2),
        Dense(units=25),
        Dense(units=1, activation='linear')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model
