# Analysis Configuration
STOCK_SYMBOL = "NVDA"  # Target stock symbol
DATASET_DIR = "Dataset"
OUTPUT_DIR = "Output"
NEWS_DATA_FILE = f"{DATASET_DIR}/news_data.csv"  # Path to your news data

UPDATE_SENTIMENT_CSV = False  # Set to True to force re-generation of sentiment data
UPDATE_STOCK_CSV = True  # Set to True to force re-fetching of stock data

# Model Parameters
SEQUENCE_LENGTH = 30  # Number of days to look back for prediction
TEST_SIZE = 0.1  # Proportion of data for testing

# ========================================================
# BASELINE MODEL CONFIGURATION
# ========================================================

BASELINE_FEATURES = ["Close", "Returns"]
BASELINE_TARGET = "Returns"

# ========================================================
# TECHNICAL MODEL CONFIGURATION
# ========================================================

TECHNICAL_FEATURES = [
    "Close",
    "SMA_50",
    "RSI",
    "MACD_line",
    "BB_width",
    "OBV",
    "Volume",
    "ATR",
    "CMF",
    "ROC",
    "Close_diff_1",
    "Close_diff_2",
    "Close_diff_5",
    "Returns",
    "Inflation_CPI",
    "Interest_Rate",
    "Unemployment_Rate",
    "VIX_Close",
]

TECHNICAL_TARGET = "Returns"

# ========================================================
# HYBRID MODEL CONFIGURATION
# ========================================================

HYBRID_FEATURES = [
    "Close",
    "SMA_50",
    "RSI",
    "MACD_line",
    "BB_width",
    "OBV",
    "Volume",
    "ATR",
    "CMF",
    "ROC",
    "Close_diff_1",
    "Close_diff_2",
    "Close_diff_5",
    "Avg_Sentiment",
    "Positive_Count",
    "Negative_Count",
    "News_Count",
    "Returns",
    "Inflation_CPI",
    "Interest_Rate",
    "Unemployment_Rate",
    "VIX_Close",
]

HYBRID_TARGET = "Returns"
