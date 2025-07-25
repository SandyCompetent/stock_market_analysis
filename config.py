# Analysis Configuration
STOCK_SYMBOL = "NVDA"  # Target stock symbol
DATASET_DIR = "Dataset"
OUTPUT_DIR = "Output"
NEWS_DATA_FILE = f"{DATASET_DIR}/news_data.csv"  # Path to your news data

UPDATE_SENTIMENT_CSV = True  # Set to True to force re-generation of sentiment data
UPDATE_STOCK_CSV = True  # Set to True to force re-fetching of stock data

# Model Parameters
SEQUENCE_LENGTH = 7  # Number of days to look back for prediction
TEST_SIZE = 0.2  # Proportion of data for testing
EPOCHS = 20  # Number of training epochs
BATCH_SIZE = 32  # Batch size for training

# Feature Lists
BASELINE_FEATURES = ["Close", "SMA_7", "RSI", "Volume", "HL_Spread", "Price_Change_Pct"]
BASELINE_TARGET = "Price_Change_Pct"

ENHANCED_FEATURES = [
    "Close",
    "SMA_7",
    "RSI",
    "Volume",
    "HL_Spread",
    "Price_Change_Pct",
    "Avg_Sentiment",
    "Sentiment_Ratio",
    "News_Count",
]

ENHANCED_TARGET = "Price_Change_Pct"
