# Advanced LSTM Stock Forecasting Notebook

## 📊 Overview

This repository now contains a comprehensive LSTM stock forecasting notebook (`USAStockMarket/analysis_using_lstm.ipynb`) that implements all the requirements specified in the issue description.

## ✅ Features Implemented

### 1. **Parameterized Analysis**
- ✅ Configurable stock ticker (AAPL, MSFT, GOOGL, etc.)
- ✅ Customizable date ranges (start and end dates)
- ✅ Adjustable model parameters (sequence length, epochs, batch size)

### 2. **Data Retrieval & Preprocessing**
- ✅ **yfinance integration** with automatic adjustment for stock splits and dividends
- ✅ **Feature engineering** with technical indicators:
  - 7-day Simple Moving Average (SMA)
  - Relative Strength Index (RSI)
  - Price change percentage
  - Volume moving average
  - High-Low spread
- ✅ **MinMaxScaler** with detailed explanation of why it's crucial for LSTMs

### 3. **Model Architecture**
- ✅ **Stacked LSTM** with dropout layers for regularization
- ✅ **Adam optimizer** and MSE loss function
- ✅ **80-20 train-test split** with discussion of walk-forward validation
- ✅ Early stopping and learning rate reduction callbacks

### 4. **Performance Analysis**
- ✅ **Baseline comparison** using naive forecast (tomorrow = today)
- ✅ **Comprehensive metrics**: RMSE, MAE, MAPE, Directional Accuracy
- ✅ **Training vs validation loss** plotting for overfitting detection
- ✅ **Statistical significance testing** (paired t-test)
- ✅ **Volatility analysis** and trend capture evaluation

### 5. **Visualization & Analysis**
- ✅ **Technical indicators charts** (Price + SMA, RSI, Volume)
- ✅ **Correlation matrix** heatmap
- ✅ **Comprehensive prediction charts**:
  - Actual vs Predicted prices
  - Prediction error distributions
  - Scatter plots for accuracy assessment
  - Rolling directional accuracy over time

### 6. **Critical Analysis**
- ✅ **Directional accuracy evaluation** (how often model predicts correct price direction)
- ✅ **Performance by market conditions** (up days vs down days)
- ✅ **Model limitations discussion**:
  - Historical bias
  - Black swan events
  - Market regime changes
  - Feature limitations
  - Transaction costs
- ✅ **Next steps for improvement**:
  - Hyperparameter tuning
  - Alternative architectures (GRU, Attention, Transformers)
  - Enhanced features (sentiment, fundamentals)
  - Ensemble methods
  - Walk-forward validation

## 🚀 Getting Started

### Prerequisites

Install required packages:
```bash
pip install numpy pandas matplotlib seaborn yfinance scikit-learn tensorflow scipy
```

**Note**: TensorFlow is required for LSTM functionality. If you encounter installation issues, try:
```bash
pip install tensorflow-cpu  # For CPU-only version
# or
pip install tensorflow-gpu  # For GPU support (requires CUDA)
```

### Usage

1. **Open the notebook**: `USAStockMarket/analysis_using_lstm.ipynb`

2. **Configure parameters** (first code cell):
   ```python
   STOCK_TICKER = 'AAPL'  # Change to any stock ticker
   START_DATE = '2020-01-01'  # Adjust date range
   END_DATE = '2024-01-01'
   ```

3. **Run all cells** to perform the complete analysis

4. **Review results**:
   - Performance metrics comparison
   - Visualization charts
   - Critical analysis insights
   - Improvement recommendations

### Example Stocks to Try
- **Tech**: AAPL, MSFT, GOOGL, TSLA, NVDA
- **Finance**: JPM, BAC, GS, WFC
- **Healthcare**: JNJ, PFE, UNH, ABBV
- **Consumer**: KO, PG, WMT, DIS

## 📈 Key Insights

The notebook provides:

1. **Model Effectiveness**: Quantitative comparison against naive baseline
2. **Directional Accuracy**: How often the model correctly predicts price direction
3. **Statistical Significance**: Whether improvements are statistically meaningful
4. **Risk Assessment**: Volatility capture and trend following analysis
5. **Practical Limitations**: Real-world considerations for trading applications

## ⚠️ Important Disclaimers

- **Educational Purpose Only**: This model is for learning and research
- **Not Financial Advice**: Do not use for actual trading without proper risk management
- **Past Performance**: Historical patterns may not predict future results
- **Risk Management**: Always implement proper position sizing and stop-losses

## 🔧 Testing

Run the dependency test:
```bash
python test_lstm_notebook.py
```

This will verify all required packages are installed and working correctly.

## 📚 Educational Value

This notebook demonstrates:
- **Time series forecasting** best practices
- **Deep learning** for financial data
- **Feature engineering** for stock analysis
- **Model evaluation** and validation techniques
- **Critical thinking** about model limitations
- **Professional presentation** of results

## 🎯 Next Steps

1. **Experiment** with different stocks and time periods
2. **Modify features** by adding more technical indicators
3. **Try different architectures** (GRU, Bidirectional LSTM)
4. **Implement walk-forward validation** for more robust testing
5. **Add fundamental data** (P/E ratios, earnings, etc.)
6. **Integrate news sentiment** analysis

---

**Created**: Advanced LSTM Stock Forecasting System  
**Status**: ✅ Complete and Ready for Use  
**Requirements**: All specifications from issue description implemented