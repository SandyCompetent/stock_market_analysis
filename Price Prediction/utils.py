import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import config as cfg
import os


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
    return start_date.strftime("%Y-%m-%d"), latest_news_date.strftime("%Y-%m-%d")


def calculate_metrics(actual, predicted, model_name):
    """Calculate comprehensive performance metrics."""
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, predicted)
    actual_direction = np.diff(actual.flatten()) > 0
    predicted_direction = np.diff(predicted.flatten()) > 0
    directional_accuracy = np.mean(actual_direction == predicted_direction) * 100

    return {
        "Model": model_name,
        "RMSE": rmse,
        "MAE": mae,
        "Directional_Accuracy": directional_accuracy,
    }


def plot_model_results(
    history, y_test, predictions, test_dates, stock_symbol, model_name
):
    """A generic function to visualize the results for any model."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle(
        f"{stock_symbol} - {model_name} Analysis", fontsize=16, fontweight="bold"
    )

    # Training history
    axes[0].plot(history.history["loss"], label="Training Loss")
    axes[0].plot(history.history["val_loss"], label="Validation Loss")
    axes[0].set_title("Model Loss During Training")
    axes[0].legend()

    # Predictions vs Actual
    axes[1].plot(test_dates, y_test, label="Actual Price", color="black")
    axes[1].plot(test_dates, predictions, label=f"{model_name} Prediction", color="red")
    axes[1].set_title("Predictions vs Actual Price")
    axes[1].legend()

    plt.tight_layout()

    safe_model_name = model_name.replace(" ", "_").replace("(", "").replace(")", "")
    save_path = os.path.join(
        cfg.OUTPUT_DIR, f"{stock_symbol}_{safe_model_name}_analysis.png"
    )
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")

    plt.show()


def plot_non_keras_results(y_test, predictions, test_dates, stock_symbol, model_name):
    """A generic function to visualize results for non-Keras models."""
    plt.figure(figsize=(12, 6))
    plt.title(f"{stock_symbol} - {model_name} Analysis", fontsize=16, fontweight="bold")
    plt.plot(test_dates, y_test, label="Actual Price", color="black")
    plt.plot(
        test_dates,
        predictions,
        label=f"{model_name} Prediction",
        color="blue",
        alpha=0.8,
    )
    plt.legend()
    plt.grid(True, alpha=0.3)

    safe_model_name = model_name.replace(" ", "_").replace("(", "").replace(")", "")
    save_path = os.path.join(
        cfg.OUTPUT_DIR, f"{stock_symbol}_{safe_model_name}_analysis.png"
    )
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")

    plt.show()


def plot_final_comparison(results, stock_symbol):
    """Creates a comparison plot for all models."""
    plt.figure(figsize=(15, 8))

    # Plot actual values first
    actual_data = results.get("Actual")
    if actual_data:
        plt.plot(
            actual_data["dates"],
            actual_data["values"],
            label="Actual Price",
            color="black",
            linewidth=2.5,
        )

    # Plot model predictions
    colors = {
        "Baseline LSTM": "blue",
        "Multi-Layer LSTM": "green",
        "Enhanced LSTM": "red",
        "Multi-Layer Enhanced LSTM": "purple",
        "Baseline SVM": "orange",
        "Enhanced SVM": "cyan",
        "ARIMA": "magenta",
    }

    # Ensure all models are plotted
    for name, data in results.items():
        if name != "Actual":
            plt.plot(
                data["dates"],
                data["values"],
                label=name,
                color=colors.get(name, "gray"),
                alpha=0.8,
            )

    plt.title(f"{stock_symbol} All Model Predictions Comparison", fontsize=16)
    plt.ylabel("Price ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save the plot
    save_path = os.path.join(
        cfg.OUTPUT_DIR, f"{stock_symbol}_all_models_comparison.png"
    )  # <-- Use cfg.OUTPUT_DIR here
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")

    plt.show()
