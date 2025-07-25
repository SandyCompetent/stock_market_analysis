import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau
import warnings
import os

# Import modules from your project
import config as cfg
import data_processing as dp
import sentiment_analysis as sa
import model as mdl
import utils as ut

# Suppress warnings
warnings.filterwarnings("ignore")
np.random.seed(42)
tf.random.set_seed(42)

# Create directories if they don't exist
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
os.makedirs(cfg.DATASET_DIR, exist_ok=True)

# --- Streamlit App ---

st.title("Stock Price Prediction with Sentiment Analysis")

st.sidebar.header("Analysis Configuration")
cfg.STOCK_SYMBOL = st.sidebar.text_input("Stock Symbol", cfg.STOCK_SYMBOL)
cfg.UPDATE_SENTIMENT_CSV = st.sidebar.checkbox(
    "Update Sentiment CSV", cfg.UPDATE_SENTIMENT_CSV
)
cfg.UPDATE_STOCK_CSV = st.sidebar.checkbox("Update Stock CSV", cfg.UPDATE_STOCK_CSV)

# --- Data Loading and Processing ---

st.header("1. Data Loading and Sentiment Analysis")

sentiment_csv_path = f"{cfg.DATASET_DIR}/{cfg.STOCK_SYMBOL}_daily_sentiment.csv"

if cfg.UPDATE_SENTIMENT_CSV or not os.path.exists(sentiment_csv_path):
    st.write("Generating new sentiment data...")
    with st.spinner("Loading news data and performing sentiment analysis..."):
        news_df = dp.load_and_analyze_news_data(cfg.NEWS_DATA_FILE, cfg.STOCK_SYMBOL)
        company_sentiment_df = sa.process_news_sentiment(news_df, cfg.STOCK_SYMBOL)
        daily_sentiment_df = sa.aggregate_daily_sentiment(company_sentiment_df)
        ut.save_dataframe(daily_sentiment_df, sentiment_csv_path)
else:
    st.write(f"Loading existing sentiment data from {sentiment_csv_path}...")
    daily_sentiment_df = pd.read_csv(
        sentiment_csv_path, index_col="Date", parse_dates=True
    )

st.write("Sentiment data ready.")
st.dataframe(daily_sentiment_df.head())

st.header("2. Stock Data and Technical Indicators")

START_DATE, END_DATE = ut.calculate_dynamic_date_range(daily_sentiment_df)

stock_filename = f"{cfg.STOCK_SYMBOL}_stock_data_{START_DATE}_to_{END_DATE}.csv"
stock_csv_path = os.path.join(cfg.DATASET_DIR, stock_filename)

if cfg.UPDATE_STOCK_CSV or not os.path.exists(stock_csv_path):
    st.write(f"Fetching new stock data from yfinance ({START_DATE} to {END_DATE})...")
    with st.spinner("Fetching stock data..."):
        stock_data = dp.fetch_stock_data(cfg.STOCK_SYMBOL, START_DATE, END_DATE)
        if stock_data is not None:
            stock_data.to_csv(stock_csv_path)
else:
    st.write(f"Loading existing stock data from {stock_csv_path}...")
    stock_data = pd.read_csv(stock_csv_path, index_col="Date", parse_dates=True)

tech_data = dp.calculate_technical_indicators(stock_data)
st.write("Technical indicators calculated.")
st.dataframe(tech_data.head())

# --- Model Training and Evaluation ---

st.header("3. LSTM Model Training and Evaluation")

# Prepare data for LSTM models
X_train_tech, X_test_tech, y_train_tech, y_test_tech, scaler_tech = (
    mdl.prepare_data_for_lstm(
        tech_data,
        cfg.BASELINE_FEATURES,
        cfg.BASELINE_TARGET,
        cfg.SEQUENCE_LENGTH,
        cfg.TEST_SIZE,
    )
)

# Single-Layer LSTM Model
st.subheader("Single-Layer LSTM Model (Baseline)")
with st.spinner("Training Single-Layer LSTM Model..."):
    baseline_model = mdl.build_single_layer_lstm(
        (X_train_tech.shape[1], X_train_tech.shape[2])
    )
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=5, min_lr=0.0001, verbose=1
    )
    baseline_history = baseline_model.fit(
        X_train_tech,
        y_train_tech,
        epochs=cfg.EPOCHS,
        batch_size=cfg.BATCH_SIZE,
        validation_split=0.2,
        callbacks=[reduce_lr],
        verbose=0,
    )
st.write("Training complete.")

# Evaluate Baseline Model
base_preds = baseline_model.predict(X_test_tech)
close_scaler = mdl.MinMaxScaler().fit(tech_data[["Close"]])
y_test_tech_scaled = close_scaler.inverse_transform(y_test_tech.reshape(-1, 1))
base_preds_scaled = close_scaler.inverse_transform(base_preds)
baseline_metrics = ut.calculate_metrics(
    y_test_tech_scaled, base_preds_scaled, "Single-Layer LSTM"
)
test_dates_tech = tech_data.index[-len(y_test_tech_scaled) :]

# --- UPDATED: Pass the figure object to st.pyplot() ---
fig_baseline = ut.plot_model_results(
    baseline_history,
    y_test_tech_scaled,
    base_preds_scaled,
    test_dates_tech,
    cfg.STOCK_SYMBOL,
    "Single-Layer LSTM",
)
st.pyplot(fig_baseline)
st.write(baseline_metrics)

# Multi-Layer LSTM Model
st.subheader("Multi-Layer LSTM Model")
with st.spinner("Training Multi-Layer LSTM Model..."):
    multi_layer_model = mdl.build_multi_layer_lstm(
        (X_train_tech.shape[1], X_train_tech.shape[2])
    )
    multi_layer_history = multi_layer_model.fit(
        X_train_tech,
        y_train_tech,
        epochs=cfg.EPOCHS,
        batch_size=cfg.BATCH_SIZE,
        validation_split=0.2,
        callbacks=[reduce_lr],
        verbose=0,
    )
st.write("Training complete.")

# Evaluate Multi-Layer Model
multi_preds = multi_layer_model.predict(X_test_tech)
multi_preds_scaled = close_scaler.inverse_transform(multi_preds)
multi_layer_metrics = ut.calculate_metrics(
    y_test_tech_scaled, multi_preds_scaled, "Multi-Layer LSTM"
)

# --- UPDATED: Pass the figure object to st.pyplot() ---
fig_multi_layer = ut.plot_model_results(
    multi_layer_history,
    y_test_tech_scaled,
    multi_preds_scaled,
    test_dates_tech,
    cfg.STOCK_SYMBOL,
    "Multi-Layer LSTM",
)
st.pyplot(fig_multi_layer)
st.write(multi_layer_metrics)

# Sentiment-Enhanced Models
st.subheader("Sentiment-Enhanced LSTM Models")
enhanced_full_data = dp.create_enhanced_dataset(tech_data, daily_sentiment_df)
X_train_enh, X_test_enh, y_train_enh, y_test_enh, scaler_enh = (
    mdl.prepare_data_for_lstm(
        enhanced_full_data,
        cfg.ENHANCED_FEATURES,
        cfg.ENHANCED_TARGET,
        cfg.SEQUENCE_LENGTH,
        cfg.TEST_SIZE,
    )
)

# Single-Layer Enhanced LSTM
st.subheader("Single-Layer LSTM Model (Enhanced with Sentiment)")
with st.spinner("Training Single-Layer Enhanced LSTM Model..."):
    enhanced_model = mdl.build_single_layer_lstm(
        (X_train_enh.shape[1], X_train_enh.shape[2])
    )
    enhanced_history = enhanced_model.fit(
        X_train_enh,
        y_train_enh,
        epochs=cfg.EPOCHS,
        batch_size=cfg.BATCH_SIZE,
        validation_split=0.2,
        callbacks=[reduce_lr],
        verbose=0,
    )
st.write("Training complete.")

# Evaluate Single-Layer Enhanced Model
enh_preds = enhanced_model.predict(X_test_enh)
y_test_enh_scaled = close_scaler.inverse_transform(y_test_enh.reshape(-1, 1))
enh_preds_scaled = close_scaler.inverse_transform(enh_preds)
enhanced_metrics = ut.calculate_metrics(
    y_test_enh_scaled, enh_preds_scaled, "Enhanced LSTM"
)
test_dates_enh = enhanced_full_data.index[-len(y_test_enh_scaled) :]

# --- UPDATED: Pass the figure object to st.pyplot() ---
fig_enhanced = ut.plot_model_results(
    enhanced_history,
    y_test_enh_scaled,
    enh_preds_scaled,
    test_dates_enh,
    cfg.STOCK_SYMBOL,
    "Enhanced LSTM",
)
st.pyplot(fig_enhanced)
st.write(enhanced_metrics)

# Multi-Layer Enhanced LSTM
st.subheader("Multi-Layer LSTM Model (Enhanced with Sentiment)")
with st.spinner("Training Multi-Layer Enhanced LSTM Model..."):
    multi_enhanced_model = mdl.build_multi_layer_lstm(
        (X_train_enh.shape[1], X_train_enh.shape[2])
    )
    multi_enhanced_history = multi_enhanced_model.fit(
        X_train_enh,
        y_train_enh,
        epochs=cfg.EPOCHS,
        batch_size=cfg.BATCH_SIZE,
        validation_split=0.2,
        callbacks=[reduce_lr],
        verbose=0,
    )
st.write("Training complete.")

# Evaluate Multi-Layer Enhanced Model
multi_enh_preds = multi_enhanced_model.predict(X_test_enh)
multi_enh_preds_scaled = close_scaler.inverse_transform(multi_enh_preds)
multi_enhanced_metrics = ut.calculate_metrics(
    y_test_enh_scaled, multi_enh_preds_scaled, "Multi-Layer Enhanced LSTM"
)

# --- UPDATED: Pass the figure object to st.pyplot() ---
fig_multi_enhanced = ut.plot_model_results(
    multi_enhanced_history,
    y_test_enh_scaled,
    multi_enh_preds_scaled,
    test_dates_enh,
    cfg.STOCK_SYMBOL,
    "Multi-Layer Enhanced LSTM",
)
st.pyplot(fig_multi_enhanced)
st.write(multi_enhanced_metrics)

# --- Final Performance Comparison ---

st.header("4. Final Performance Comparison")

all_metrics_df = pd.DataFrame(
    [
        baseline_metrics,
        enhanced_metrics,
        multi_layer_metrics,
        multi_enhanced_metrics,
    ]
).round(4)

st.dataframe(all_metrics_df)

plot_data = {
    "Actual": {"dates": test_dates_tech, "values": y_test_tech_scaled},
    "Baseline LSTM": {"dates": test_dates_tech, "values": base_preds_scaled},
    "Multi-Layer LSTM": {"dates": test_dates_tech, "values": multi_preds_scaled},
    "Enhanced LSTM": {"dates": test_dates_enh, "values": enh_preds_scaled},
    "Multi-Layer Enhanced LSTM": {
        "dates": test_dates_enh,
        "values": multi_enh_preds_scaled,
    },
}

# --- UPDATED: Pass the figure object to st.pyplot() ---
fig_final = ut.plot_final_comparison(plot_data, cfg.STOCK_SYMBOL)
st.pyplot(fig_final)
