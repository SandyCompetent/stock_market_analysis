import pandas as pd
import yfinance as yf
import numpy as np


# import kagglehub
# from kagglehub import KaggleDatasetAdapter


def load_and_analyze_news_data(file_path, target_symbol):
    print("Loading and selecting news data...")
    # try:
    #     df = kagglehub.load_dataset(
    #         KaggleDatasetAdapter.PANDAS,
    #         "miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests",
    #         "analyst_ratings_processed.csv"
    #     )
    # except Exception as e:
    #     print(f"⚠️ ERROR: Failed to load dataset from Kaggle: {e}")
    #     return None

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
    df = data.copy()
    df["SMA_7"] = df["Close"].rolling(window=7).mean()
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))
    df["Price_Change_Pct"] = df["Close"].pct_change()
    df["Volume_MA"] = df["Volume"].rolling(window=7).mean()
    df["HL_Spread"] = (df["High"] - df["Low"]) / df["Close"]
    return df.dropna()


def create_enhanced_dataset(stock_data_with_indicators, daily_sentiment_df):
    print("\\n🔗 Merging sentiment and technical data...")
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
