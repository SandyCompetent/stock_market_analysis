import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import config as cfg
import os
import datetime

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
from sklearn.base import BaseEstimator, RegressorMixin
from scipy import stats


def format_runtime(total_seconds):
    """Formats seconds into a human-readable string (D H M S)."""
    # Create a timedelta object
    td = datetime.timedelta(seconds=total_seconds)

    # Extract days, and the remaining seconds
    days = td.days
    remaining_seconds = td.seconds

    # Calculate hours, minutes, and seconds from the remainder
    hours, rem = divmod(remaining_seconds, 3600)
    minutes, seconds = divmod(rem, 60)

    # Add microseconds for fractional second precision
    seconds += td.microseconds / 1_000_000

    # Build the output string
    parts = []
    if days > 0:
        parts.append(f"{days} day{'s' if days != 1 else ''}")
    if hours > 0:
        parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
    if minutes > 0:
        parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
    if seconds > 0 or not parts:  # Always show seconds if it's the only unit
        parts.append(f"{seconds:.2f} seconds")

    return " ".join(parts)


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


def calculate_metrics(actual, predicted, model_name, train_actual):
    """Calculate comprehensive performance metrics."""

    # Ensure inputs are numpy arrays for calculations
    actual = np.asarray(actual).flatten()
    predicted = np.asarray(predicted).flatten()
    train_actual = np.asarray(train_actual).flatten()

    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, predicted)

    # Mean Absolute Percentage Error (MAPE) calculation (adding a small epsilon to avoid division by zero)
    mape = np.mean(np.abs((actual - predicted) / (actual + 1e-8))) * 100

    # R-squared calculation
    r2 = r2_score(actual, predicted)

    # Directional accuracy
    actual_direction = np.diff(actual.flatten()) > 0
    predicted_direction = np.diff(predicted.flatten()) > 0
    directional_accuracy = np.mean(actual_direction == predicted_direction) * 100

    # Mean Absolute Scaled Error (MASE) calculation
    # The denominator is the MAE of the naive forecast on the training set
    mae_naive_train = np.mean(np.abs(np.diff(train_actual)))
    mase = mae / (mae_naive_train + 1e-8)  # add epsilon to avoid division by zero

    return {
        "Model": model_name,
        "RMSE": rmse,
        "MAE": mae,
        "MAPE (%)": mape,
        "R-squared": r2,
        "Directional_Accuracy": directional_accuracy,
        "MASE": mase,
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
    )  # <-- Use cfg.OUTPUT_DIR here
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


def plot_final_comparison(results, stock_symbol, title):
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
        "Actual": "black",
        "Single-Layer LSTM": "blue",
        "Multi-Layer LSTM": "green",
        "Enhanced LSTM": "red",
        "Multi-Layer Enhanced LSTM": "purple",
        "Baseline GRU": "brown",
        "Enhanced GRU": "pink",
        "SVM": "orange",
        "Enhanced SVM": "cyan",
        "ARIMA": "magenta",
    }
    for name, data in results.items():
        if name != "Actual":
            plt.plot(
                data["dates"],
                data["values"],
                label=name,
                color=colors.get(name, "gray"),
                alpha=0.8,
            )

    plt.title(title, fontsize=16)
    plt.ylabel("Daily Return")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save the plot
    save_path = os.path.join(
        cfg.OUTPUT_DIR, f"{stock_symbol}_all_models_comparison.png"
    )  # <-- Use cfg.OUTPUT_DIR here
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")

    plt.show()


def plot_residuals(y_test, predictions, test_dates, stock_symbol, model_name):
    """
    Plots the residuals (errors) of the model's predictions over time.
    """
    residuals = y_test.flatten() - predictions.flatten()

    plt.figure(figsize=(14, 6))
    plt.plot(
        test_dates,
        residuals,
        label="Residuals (Actual - Predicted)",
        color="purple",
        alpha=0.8,
    )
    plt.axhline(y=0, color="r", linestyle="--", label="Zero Error")

    plt.title(f"{stock_symbol} - {model_name} Prediction Residuals", fontsize=16)
    plt.xlabel("Date")
    plt.ylabel("Prediction Error (Return)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    safe_model_name = model_name.replace(" ", "_").replace("(", "").replace(")", "")
    save_path = os.path.join(
        cfg.OUTPUT_DIR, f"{stock_symbol}_{safe_model_name}_residuals.png"
    )
    plt.savefig(save_path)
    print(f"Residuals plot saved to {save_path}")

    plt.show()


def plot_enhanced_diagnostics(
    y_test, predictions, test_dates, stock_symbol, model_name
):
    """
    Creates enhanced diagnostic plots for model evaluation:
    1. Residuals vs Predicted Values (to check for heteroscedasticity)
    2. Actual vs Predicted Values (scatter plot)
    3. Distribution of Residuals (histogram with normal distribution overlay)
    """
    # Ensure inputs are numpy arrays and flatten them
    y_test = np.asarray(y_test).flatten()
    predictions = np.asarray(predictions).flatten()
    residuals = y_test - predictions

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        f"{stock_symbol} - {model_name} Enhanced Diagnostic Analysis",
        fontsize=16,
        fontweight="bold",
    )

    # 1. Residuals vs Predicted Values
    axes[0, 0].scatter(predictions, residuals, alpha=0.6, color="blue", s=30)
    axes[0, 0].axhline(y=0, color="red", linestyle="--", linewidth=2)
    axes[0, 0].set_xlabel("Predicted Values")
    axes[0, 0].set_ylabel("Residuals")
    axes[0, 0].set_title(
        "Residuals vs Predicted Values\n(Check for Heteroscedasticity)"
    )
    axes[0, 0].grid(True, alpha=0.3)

    # Add trend line to residuals vs predicted
    z = np.polyfit(predictions, residuals, 1)
    p = np.poly1d(z)
    axes[0, 0].plot(predictions, p(predictions), "r--", alpha=0.8, linewidth=1)

    # 2. Actual vs Predicted Values (Scatter Plot)
    axes[0, 1].scatter(y_test, predictions, alpha=0.6, color="green", s=30)

    # Perfect prediction line (y=x)
    min_val = min(min(y_test), min(predictions))
    max_val = max(max(y_test), max(predictions))
    axes[0, 1].plot(
        [min_val, max_val],
        [min_val, max_val],
        "r--",
        linewidth=2,
        label="Perfect Prediction",
    )

    axes[0, 1].set_xlabel("Actual Values")
    axes[0, 1].set_ylabel("Predicted Values")
    axes[0, 1].set_title("Actual vs Predicted Values")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Calculate and display R² on the plot
    r2 = r2_score(y_test, predictions)
    axes[0, 1].text(
        0.05,
        0.95,
        f"R² = {r2:.4f}",
        transform=axes[0, 1].transAxes,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    # 3. Distribution of Residuals (Histogram with Normal Distribution Overlay)
    axes[1, 0].hist(
        residuals,
        bins=30,
        density=True,
        alpha=0.7,
        color="skyblue",
        edgecolor="black",
        label="Residuals",
    )

    # Overlay normal distribution
    mu, sigma = stats.norm.fit(residuals)
    x = np.linspace(residuals.min(), residuals.max(), 100)
    axes[1, 0].plot(
        x,
        stats.norm.pdf(x, mu, sigma),
        "r-",
        linewidth=2,
        label=f"Normal (μ={mu:.4f}, σ={sigma:.4f})",
    )

    axes[1, 0].set_xlabel("Residuals")
    axes[1, 0].set_ylabel("Density")
    axes[1, 0].set_title("Distribution of Residuals\n(Check for Normality)")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Q-Q Plot for Normality Check
    stats.probplot(residuals, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title("Q-Q Plot\n(Normality Check)")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save the plot
    safe_model_name = model_name.replace(" ", "_").replace("(", "").replace(")", "")
    save_path = os.path.join(
        cfg.OUTPUT_DIR, f"{stock_symbol}_{safe_model_name}_enhanced_diagnostics.png"
    )
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Enhanced diagnostics plot saved to {save_path}")

    plt.show()

    # Print statistical summary
    print(f"\n=== {model_name} Diagnostic Summary ===")
    print(f"Residuals Mean: {np.mean(residuals):.6f}")
    print(f"Residuals Std: {np.std(residuals):.6f}")
    print(f"Residuals Skewness: {stats.skew(residuals):.4f}")
    print(f"Residuals Kurtosis: {stats.kurtosis(residuals):.4f}")

    # Shapiro-Wilk test for normality (if sample size is reasonable)
    if len(residuals) <= 5000:
        shapiro_stat, shapiro_p = stats.shapiro(residuals)
        print(
            f"Shapiro-Wilk Test: statistic={shapiro_stat:.4f}, p-value={shapiro_p:.4f}"
        )
        if shapiro_p > 0.05:
            print("✓ Residuals appear to be normally distributed (p > 0.05)")
        else:
            print("✗ Residuals may not be normally distributed (p ≤ 0.05)")

    # Durbin-Watson test for autocorrelation
    dw_stat = np.sum(np.diff(residuals) ** 2) / np.sum(residuals**2)
    print(f"Durbin-Watson Statistic: {dw_stat:.4f}")
    if 1.5 < dw_stat < 2.5:
        print("✓ No significant autocorrelation detected")
    else:
        print("✗ Potential autocorrelation in residuals")


def plot_model_comparison_metrics(results_df, stock_symbol):
    """
    Creates comprehensive bar charts comparing performance metrics across all models.
    """
    # Prepare the data
    models = results_df["Model"].values
    metrics = ["RMSE", "MAE", "MAPE (%)", "R-squared", "Directional_Accuracy", "MASE"]

    # Create subplots for different metrics
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(
        f"{stock_symbol} - Model Performance Comparison", fontsize=16, fontweight="bold"
    )

    # Flatten axes for easier iteration
    axes = axes.flatten()

    # Color palette for models
    colors = plt.cm.Set3(np.linspace(0, 1, len(models)))

    for i, metric in enumerate(metrics):
        values = results_df[metric].values

        # Create bar plot
        bars = axes[i].bar(
            range(len(models)), values, color=colors, alpha=0.8, edgecolor="black"
        )

        # Customize the plot
        axes[i].set_title(f"{metric}", fontsize=12, fontweight="bold")
        axes[i].set_xticks(range(len(models)))
        axes[i].set_xticklabels(models, rotation=45, ha="right")
        axes[i].grid(True, alpha=0.3, axis="y")

        # Add value labels on bars
        for j, (bar, value) in enumerate(zip(bars, values)):
            height = bar.get_height()
            axes[i].text(
                bar.get_x() + bar.get_width() / 2.0,
                height + height * 0.01,
                (
                    f"{value:.4f}"
                    if metric != "MAPE (%)" and metric != "Directional_Accuracy"
                    else (
                        f"{value:.2f}%"
                        if metric == "MAPE (%)" or metric == "Directional_Accuracy"
                        else f"{value:.4f}"
                    )
                ),
                ha="center",
                va="bottom",
                fontsize=8,
            )

        # Highlight best performing model
        if metric in ["R-squared", "Directional_Accuracy"]:
            best_idx = np.argmax(values)
        else:
            best_idx = np.argmin(values)

        bars[best_idx].set_color("gold")
        bars[best_idx].set_edgecolor("red")
        bars[best_idx].set_linewidth(2)

    plt.tight_layout()

    # Save the plot
    save_path = os.path.join(
        cfg.OUTPUT_DIR, f"{stock_symbol}_model_comparison_metrics.png"
    )
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Model comparison metrics plot saved to {save_path}")

    plt.show()


def plot_top_models_comparison(results_dict, stock_symbol, top_n=4):
    """
    Creates a comprehensive line plot showing predictions of top-performing models
    against actual stock prices.
    """
    plt.figure(figsize=(16, 10))

    # Plot actual values first
    actual_data = results_dict.get("Actual")
    if actual_data:
        plt.plot(
            actual_data["dates"],
            actual_data["values"],
            label="Actual Price",
            color="black",
            linewidth=3,
            alpha=0.8,
        )

    # Define colors for different models
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    # Plot top models (excluding 'Actual')
    model_names = [name for name in results_dict.keys() if name != "Actual"]

    for i, (name, data) in enumerate(list(results_dict.items())[: top_n + 1]):
        if name != "Actual":
            plt.plot(
                data["dates"],
                data["values"],
                label=name,
                color=colors[i % len(colors)],
                linewidth=2,
                alpha=0.8,
            )

    plt.title(
        f"{stock_symbol} - Top {top_n} Model Predictions vs Actual Prices",
        fontsize=16,
        fontweight="bold",
    )
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Daily Return", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)

    # Add performance statistics as text box
    textstr = (
        f'Comparison Period: {len(actual_data["dates"]) if actual_data else "N/A"} days'
    )
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
    plt.text(
        0.02,
        0.98,
        textstr,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=props,
    )

    plt.tight_layout()

    # Save the plot
    save_path = os.path.join(
        cfg.OUTPUT_DIR, f"{stock_symbol}_top_models_comparison.png"
    )
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Top models comparison plot saved to {save_path}")

    plt.show()
