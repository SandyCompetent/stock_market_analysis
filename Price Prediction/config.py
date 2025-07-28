# Analysis Configuration
STOCK_SYMBOL = "NVDA"  # Target stock symbol
DATASET_DIR = "Dataset"
OUTPUT_DIR = "Output"
NEWS_DATA_FILE = f"{DATASET_DIR}/news_data.csv"  # Path to your news data

UPDATE_SENTIMENT_CSV = True  # Set to True to force re-generation of sentiment data
UPDATE_STOCK_CSV = True  # Set to True to force re-fetching of stock data

# Model Parameters
SEQUENCE_LENGTH = 30  # Number of days to look back for prediction
TEST_SIZE = 0.2  # Proportion of data for testing
EPOCHS = 100  # Number of training epochs
BATCH_SIZE = 100  # Batch size for training

BASELINE_FEATURES = [
    "Close",
    "SMA_50",
    "RSI",
    "MACD_line",
    "BB_width",
    "OBV",
    "Volume"
]

BASELINE_TARGET = "Close"

ENHANCED_FEATURES = [
    "Close",
    "SMA_50",
    "RSI",
    "MACD_line",
    "BB_width",
    "OBV",
    "Volume",
    "Avg_Sentiment",
    "Sentiment_Ratio",
    "News_Count",
]
ENHANCED_TARGET = "Close"
