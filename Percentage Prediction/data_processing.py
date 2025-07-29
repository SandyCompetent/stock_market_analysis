import pandas as pd
import yfinance as yf
import numpy as np


def load_and_analyze_news_data(file_path, target_symbol):
    print("Loading and selecting news data...")
    df = pd.read_csv(file_path)
    df["Date"] = pd.to_datetime(df["date"], format="mixed", utc=True, errors="coerce")
    df = df.dropna(subset=["Date"])

    if target_symbol in df["stock"].unique():
        article_count = df[df["stock"] == target_symbol].shape[0]
        print(f"🎯 Selected target stock: {target_symbol} ({article_count} articles)")
        return df
    else:
        print(f"⚠️ ERROR: Target stock '{target_symbol}' not found.")
        return None


def fetch_stock_data(ticker, start_date, end_date):
    print(f"\\n📈 Fetching stock data for {ticker}...")
    try:
        data = yf.Ticker(ticker).history(start=start_date, end=end_date, interval="1d")
        if data.empty:
            raise ValueError(f"No data found for ticker {ticker}")
        print(f"Successfully fetched {len(data)} days of data.")
        return data
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None


def calculate_technical_indicators(data):
    """
    Calculates technical indicators for the given stock data using pandas and numpy.

    Args:
        data (pd.DataFrame): DataFrame with stock data.

    Returns:
        pd.DataFrame: DataFrame with calculated technical indicators.
    """
    df = data.copy()

    # Calculate Percentage Change in Price
    df["Pct_Change"] = df["Close"].pct_change() * 100

    # 1. Moving Averages (SMA & EMA)
    df["SMA_50"] = df["Close"].rolling(window=50).mean()
    df["SMA_200"] = df["Close"].rolling(window=200).mean()
    df["EMA_50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["EMA_200"] = df["Close"].ewm(span=200, adjust=False).mean()

    # 2. Moving Average Convergence Divergence (MACD)
    ema_12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD_line"] = ema_12 - ema_26
    df["MACD_signal_line"] = df["MACD_line"].ewm(span=9, adjust=False).mean()
    df["MACD_histogram"] = df["MACD_line"] - df["MACD_signal_line"]

    # 3. Average Directional Index (ADX)
    period = 14
    tr = pd.DataFrame()
    tr["h-l"] = df["High"] - df["Low"]
    tr["h-pc"] = abs(df["High"] - df["Close"].shift(1))
    tr["l-pc"] = abs(df["Low"] - df["Close"].shift(1))
    tr["tr"] = tr[["h-l", "h-pc", "l-pc"]].max(axis=1)
    atr = tr["tr"].ewm(span=period, adjust=False).mean()

    df["+DI"] = 100 * (
        df["High"]
        .diff()
        .where(df["High"].diff() > df["Low"].diff(), 0)
        .ewm(alpha=1 / period, adjust=False)
        .mean()
        / atr
    )
    df["-DI"] = 100 * (
        df["Low"]
        .diff()
        .where(df["Low"].diff() > df["High"].diff(), 0)
        .ewm(alpha=1 / period, adjust=False)
        .mean()
        / atr
    )

    dx = 100 * abs(df["+DI"] - df["-DI"]) / (df["+DI"] + df["-DI"])
    df["ADX"] = dx.ewm(alpha=1 / period, adjust=False).mean()

    # 4. Relative Strength Index (RSI)
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).ewm(span=14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(span=14, adjust=False).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # 5. Stochastic Oscillator
    low_14 = df["Low"].rolling(window=14).min()
    high_14 = df["High"].rolling(window=14).max()
    df["%K"] = 100 * ((df["Close"] - low_14) / (high_14 - low_14))
    df["%D"] = df["%K"].rolling(window=3).mean()

    # 6. Bollinger Bands
    df["BB_middle"] = df["Close"].rolling(window=20).mean()
    std_dev = df["Close"].rolling(window=20).std()
    df["BB_upper"] = df["BB_middle"] + (std_dev * 2)
    df["BB_lower"] = df["BB_middle"] - (std_dev * 2)
    df["BB_width"] = df["BB_upper"] - df["BB_lower"]

    # 7. On-Balance Volume (OBV)
    df["OBV"] = (np.sign(df["Close"].diff()) * df["Volume"]).fillna(0).cumsum()

    # Drop rows with NaN values created by rolling windows
    df = df.dropna()

    return df


def create_enhanced_dataset(stock_data_with_indicators, daily_sentiment_df):
    print("\n🔗 Merging sentiment and technical data...")
    if daily_sentiment_df is None or daily_sentiment_df.empty:
        return None

    enhanced_stock_data = stock_data_with_indicators.copy()
    enhanced_stock_data["Date"] = pd.to_datetime(enhanced_stock_data.index.date)
    enhanced_stock_data = enhanced_stock_data.set_index("Date")

    merged_data = enhanced_stock_data.join(daily_sentiment_df, how="left")

    sentiment_cols = ["Avg_Sentiment", "Total_Sentiment", "News_Count"]
    merged_data[sentiment_cols] = merged_data[sentiment_cols].fillna(0)

    # Simple sentiment ratio for modeling
    merged_data["Sentiment_Ratio"] = (
        merged_data["Avg_Sentiment"] * merged_data["News_Count"]
    ).fillna(0)

    print(f"Enhanced dataset shape: {merged_data.shape}")
    return merged_data
