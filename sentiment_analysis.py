import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm.auto import tqdm


class SentimentAnalyzer:
    """
    A class to efficiently perform sentiment analysis using FinBERT,
    with support for batching.
    """

    def __init__(self, model_name="ProsusAI/finbert"):
        print(f"Initializing sentiment analyzer with model: {model_name}")
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Using device: {self.device}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.to(self.device)
            self.sentiment_map = self.model.config.id2label
            self.sentiment_map = self.model.config.id2label
            self.label2id = {v.lower(): k for k, v in self.sentiment_map.items()}
            self.pos_id = self.label2id.get("positive")
            self.neg_id = self.label2id.get("negative")
            if self.pos_id is None or self.neg_id is None:
                raise KeyError(
                    "Model labels do not include 'positive' and/or 'negative'."
                )
            print("Analyzer initialized successfully.")
        except Exception as e:
            print(f"Error initializing model: {e}")
            raise

    def analyze(self, text_list, batch_size=32):
        all_sentiments, all_confidence_scores, all_single_scores = [], [], []
        print(f"Analyzing {len(text_list)} headlines in batches of {batch_size}...")
        for i in tqdm(range(0, len(text_list), batch_size)):
            batch = list(text_list[i: i + batch_size])
            cleaned_batch = [str(text) if text is not None else "" for text in batch]
            try:
                inputs = self.tokenizer(
                    cleaned_batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                )
                inputs = {key: val.to(self.device) for key, val in inputs.items()}
                with torch.no_grad():
                    outputs = self.model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
                confidence_scores, predictions = torch.max(probabilities, dim=1)
                sentiments = [self.sentiment_map[pred.item()] for pred in predictions]
                single_scores = (
                        probabilities[:, self.pos_id] - probabilities[:, self.neg_id]
                )
                all_sentiments.extend(sentiments)
                all_confidence_scores.extend(confidence_scores.cpu().tolist())
                all_single_scores.extend(single_scores.cpu().tolist())
            except Exception as e:
                print(f"Error analyzing batch at index {i}: {e}")
                all_sentiments.extend(["Unknown"] * len(batch))
                all_confidence_scores.extend([0.0] * len(batch))
                all_single_scores.extend([0.0] * len(batch))
        return all_sentiments, all_confidence_scores, all_single_scores


def process_news_sentiment(news_df, target_stock):
    print(f"\\n📰 Processing news sentiment for {target_stock}...")
    company_news = news_df[news_df["stock"] == target_stock].copy()
    if company_news.empty:
        print(f"No news found for {target_stock}")
        return None

    company_news["date"] = pd.to_datetime(
        company_news["date"], utc=True, errors="coerce"
    )
    company_news = company_news.dropna(subset=["date"])
    company_news["Date"] = company_news["date"].dt.date

    analyzer = SentimentAnalyzer()
    (
        company_news["Sentiment"],
        company_news["Confidence"],
        company_news["Sentiment_Score"],
    ) = analyzer.analyze(company_news["title"], batch_size=32)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\\nSentiment distribution:\\n", company_news["Sentiment"].value_counts())
    return company_news


def aggregate_daily_sentiment(sentiment_df):
    if sentiment_df is None or sentiment_df.empty:
        return None
    print("\n📊 Aggregating daily sentiment scores...")

    # Create dummy variables for one-hot encoding of sentiment
    sentiment_dummies = pd.get_dummies(sentiment_df['Sentiment'], prefix='Sentiment')
    sentiment_df = pd.concat([sentiment_df, sentiment_dummies], axis=1)

    # Ensure all possible sentiment columns exist, filling missing ones with 0
    expected_sentiment_cols = ['Sentiment_Positive', 'Sentiment_Negative', 'Sentiment_Neutral']
    for col in expected_sentiment_cols:
        if col not in sentiment_df.columns:
            sentiment_df[col] = 0

    # Define aggregations
    agg_functions = {
        'Sentiment_Score': ['mean', 'sum'],
        'Sentiment_Positive': ['sum'],
        'Sentiment_Negative': ['sum'],
        'Sentiment_Neutral': ['sum']
    }

    # Group by date and aggregate
    daily_sentiment = sentiment_df.groupby('Date').agg(agg_functions).round(4)

    # Flatten the multi-level column names
    daily_sentiment.columns = [
        'Avg_Sentiment', 'Total_Sentiment',
        'Positive_Count', 'Negative_Count', 'Neutral_Count'
    ]

    # Also add the total news count for the day
    daily_sentiment['News_Count'] = sentiment_df.groupby('Date')['title'].count()

    return daily_sentiment
