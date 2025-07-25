import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error


def save_dataframe(df: pd.DataFrame, file_path: str):
    try:
        df.to_csv(file_path)
        print(f"DataFrame saved to {file_path}")
    except Exception as e:
        print(f"Error: Failed to save DataFrame: {e}")


def calculate_dynamic_date_range(daily_sentiment_df):
    """Calculates start and end dates based on news data availability."""
    if daily_sentiment_df is None or daily_sentiment_df.empty:
        return None, None
    earliest_news_date = pd.to_datetime(daily_sentiment_df.index.min())
    latest_news_date = pd.to_datetime(daily_sentiment_df.index.max())
    start_date = earliest_news_date - pd.DateOffset(years=1)
    return start_date.strftime('%Y-%m-%d'), latest_news_date.strftime('%Y-%m-%d')


def calculate_metrics(actual, predicted, model_name):
    """Calculate comprehensive performance metrics."""
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, predicted)
    actual_direction = np.diff(actual.flatten()) > 0
    predicted_direction = np.diff(predicted.flatten()) > 0
    directional_accuracy = np.mean(actual_direction == predicted_direction) * 100

    return {
        'Model': model_name,
        'RMSE': rmse,
        'MAE': mae,
        'Directional_Accuracy': directional_accuracy
    }


def plot_baseline_results(history, y_test, predictions, test_dates, stock_symbol):
    """Visualizes the results for the baseline model."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle(f'{stock_symbol} Baseline LSTM Analysis', fontsize=16, fontweight='bold')

    # Training history
    axes[0].plot(history.history['loss'], label='Training Loss')
    axes[0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0].set_title('Model Loss During Training')
    axes[0].legend()

    # Predictions vs Actual
    axes[1].plot(test_dates, y_test, label='Actual Price', color='black')
    axes[1].plot(test_dates, predictions, label='LSTM Prediction', color='red')
    axes[1].set_title('Predictions vs Actual Price')
    axes[1].legend()

    plt.tight_layout()
    plt.show()


def plot_enhanced_results(history, y_test, predictions, test_dates, stock_symbol):
    """Visualizes the results for the sentiment-enhanced model."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle(f'{stock_symbol} Sentiment-Enhanced LSTM Analysis', fontsize=16, fontweight='bold')

    # 1. Training history
    axes[0].plot(history.history['loss'], label='Training Loss', color='red')
    axes[0].plot(history.history['val_loss'], label='Validation Loss', color='orange')
    axes[0].set_title('Enhanced Model Loss During Training')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (MSE)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2. Predictions vs Actual
    axes[1].plot(test_dates, y_test, label='Actual Price', color='black', linewidth=2)
    axes[1].plot(test_dates, predictions, label='Enhanced LSTM Prediction', color='red', alpha=0.8)
    axes[1].set_title('Enhanced Model Predictions vs Actual Price')
    axes[1].set_ylabel('Price ($)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()


def plot_comparison_results(y_test_base, base_preds, y_test_enh, enh_preds, test_dates_base, test_dates_enh,
                            stock_symbol):
    """Creates a comprehensive visualization comparing both models."""
    plt.figure(figsize=(14, 7))
    plt.plot(test_dates_base, y_test_base, label='Actual Price', color='black', linewidth=2)
    plt.plot(test_dates_base, base_preds, label='Baseline LSTM', color='blue', alpha=0.8)
    plt.plot(test_dates_enh, enh_preds, label='Enhanced LSTM', color='red', alpha=0.8)
    plt.title(f'{stock_symbol} LSTM Models Comparison: Baseline vs Sentiment-Enhanced', fontsize=16)
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
