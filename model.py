import numpy as np
import warnings

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.sm_exceptions import ConvergenceWarning


# ======================================================
# PREPARING DATA FOR LSTM
# ======================================================


def prepare_data_for_lstm(
    data, feature_columns, target_column, sequence_length, test_size
):
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
        X_train.append(train_scaled[i - sequence_length : i, :])
        y_train.append(train_scaled[i, target_idx])

    inputs = np.concatenate((train_scaled[-sequence_length:], test_scaled), axis=0)
    X_test, y_test = [], []
    for i in range(sequence_length, len(inputs)):
        X_test.append(inputs[i - sequence_length : i, :])
        y_test.append(inputs[i, target_idx])

    return (
        np.array(X_train),
        np.array(X_test),
        np.array(y_train),
        np.array(y_test),
        scaler,
    )


# ======================================================
# SINGLE-LAYER LSTM
# ======================================================


def build_single_layer_lstm(hp, input_shape):
    """Builds a single-layer LSTM model."""
    model = Sequential()

    # Define the search space for hyperparameters
    hp_units = hp.Int("units", min_value=32, max_value=256, step=32)
    hp_learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])
    hp_dropout = hp.Float("dropout", min_value=0.1, max_value=0.5, step=0.1)

    # Add the first LSTM layer with the dynamic input_shape
    model.add(LSTM(units=hp_units, input_shape=input_shape))
    model.add(Dropout(rate=hp_dropout))
    model.add(Dense(units=1, activation="linear"))

    model.compile(
        optimizer=Adam(learning_rate=hp_learning_rate),
        loss="mean_squared_error",
    )
    return model


def build_multi_layer_lstm(hp, input_shape):
    """Builds a stacked, multi-layer LSTM model for hyperparameter tuning."""
    model = Sequential()

    # Tune the number of units in the first LSTM layer
    hp_units_1 = hp.Int("units_1", min_value=64, max_value=256, step=64)
    # Tune the number of units in the second LSTM layer
    hp_units_2 = hp.Int("units_2", min_value=32, max_value=128, step=32)
    # Tune the dropout rate
    hp_dropout = hp.Float("dropout", min_value=0.1, max_value=0.5, step=0.1)
    # Tune the learning rate
    hp_learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])

    model.add(
        LSTM(
            units=hp_units_1,
            return_sequences=True,  # Important: This must be True for stacking
            input_shape=input_shape,
        )
    )
    model.add(Dropout(rate=hp_dropout))

    model.add(LSTM(units=hp_units_2, return_sequences=False))
    model.add(Dropout(rate=hp_dropout))

    # A dense layer before the output can sometimes help capture more complex patterns
    model.add(Dense(units=25))
    model.add(Dense(units=1, activation="linear"))

    model.compile(
        optimizer=Adam(learning_rate=hp_learning_rate),
        loss="mean_squared_error",
    )
    return model


# ======================================================
# GRU
# ======================================================


def build_gru(hp, input_shape):
    """
    Creates a GRU model for hyperparameter tuning.
    """
    model = Sequential()

    # Define the search space for hyperparameters
    hp_units_1 = hp.Int("units_1", min_value=64, max_value=256, step=64)
    hp_units_2 = hp.Int("units_2", min_value=32, max_value=128, step=32)
    hp_dropout = hp.Float("dropout", min_value=0.1, max_value=0.5, step=0.1)
    hp_learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])

    model.add(
        GRU(
            units=hp_units_1,
            return_sequences=True,
            input_shape=input_shape,
        )
    )
    model.add(Dropout(rate=hp_dropout))
    model.add(GRU(units=hp_units_2, return_sequences=False))
    model.add(Dropout(rate=hp_dropout))
    model.add(Dense(25))
    model.add(Dense(1))

    model.compile(
        optimizer=Adam(learning_rate=hp_learning_rate), loss="mean_squared_error"
    )
    return model


# ======================================================
# SVM
# ======================================================


def build_and_train_svm(X_train, y_train):
    """Builds and finds the best SVM model using GridSearchCV."""
    print("Building and tuning SVM model with GridSearchCV...")

    # Define the grid of parameters to search
    # A smaller grid is used here to keep tuning time reasonable.
    # You can expand this grid for a more exhaustive search.
    param_grid = {
        "C": [1, 10, 100],
        "gamma": ["scale", 0.1, 0.01],
        "kernel": ["rbf"],
        "epsilon": [0.1, 0.05],
    }

    # Use SVR for regression
    svr = SVR()

    # Set up GridSearchCV to test all parameter combinations.
    # cv=3 means 3-fold cross-validation.
    grid_search = GridSearchCV(
        estimator=svr, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2
    )

    # Run the grid search
    grid_search.fit(X_train, y_train)

    print(f"Best SVM parameters found: {grid_search.best_params_}")
    print("SVM tuning complete.")

    # Return the best model found by the search
    return grid_search.best_estimator_


# ======================================================
# ARIMA
# ======================================================


def find_best_arima_order(train_data):
    """
    Iterates through p, d, q ranges to find the best ARIMA order based on AIC.
    """
    print("Searching for the best ARIMA order...")
    # A smaller range is used to keep search time manageable.
    p_values = range(0, 6)  # Autoregressive order
    d_values = range(0, 2)  # Differencing order
    q_values = range(0, 3)  # Moving average order
    best_aic, best_order = float("inf"), None

    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    for p in p_values:
        for d in d_values:
            for q in q_values:
                try:
                    model = ARIMA(train_data, order=(p, d, q))
                    fitted_model = model.fit()
                    if fitted_model.aic < best_aic:
                        best_aic = fitted_model.aic
                        best_order = (p, d, q)
                except Exception as e:
                    continue  # Skip orders that cause errors

    print(f"Best ARIMA order found: {best_order} with AIC: {best_aic:.2f}")
    warnings.resetwarnings()  # Reset warnings to default
    return best_order


def build_and_train_arima(train_data, order):
    """Builds and trains a simple ARIMA model with a given order."""
    arima_model = ARIMA(train_data, order=order)
    fitted_model = arima_model.fit()
    return fitted_model
