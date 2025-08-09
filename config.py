# Analysis Configuration
STOCK_SYMBOL = "NVDA"  # Target stock symbol
DATASET_DIR = "Dataset"
OUTPUT_DIR = "Output_NVDA"
NEWS_DATA_FILE = f"{DATASET_DIR}/news_data.csv"
LOCAL_STOCK_FILE_PATH = f"{DATASET_DIR}/NVDA_stock_data.csv"

UPDATE_SENTIMENT_CSV = True  # Set to True to force re-generation of sentiment data
UPDATE_STOCK_CSV = True  # Set to True to force re-fetching of stock data
LOAD_LOCAL_STOCK_FILE = True
OVERWRITE_TUNNER_RESULT = True

RESAMPLE_DATA = False

# Model Parameters
# 'D'  - Calendar day
# 'W'  - Weekly (e.g., 'W-FRI' for Friday-ending weeks)
# 'M'  - Month end
# 'Q'  - Quarter end
# 'Y'
#
# --- Time-Based Frequencies ---
# 'H'  - Hourly
# 'T'  - Minutely

RESAMPLE_FREQUENCY = "1D"
SEQUENCE_LENGTH = 90  # Number of days to look back for prediction
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
