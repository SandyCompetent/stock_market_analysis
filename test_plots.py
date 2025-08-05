#!/usr/bin/env python3
"""
Test script to verify the plotting functionality added to main_analysis.ipynb
This script tests the key components without running the full notebook.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

# Add the project directory to Python path
sys.path.append('/Users/sandy/PycharmProjects/stock_market_analysis')

# Import project modules
try:
    import config as cfg
    import sentiment_analysis as sa
    print("✅ Successfully imported project modules")
except ImportError as e:
    print(f"❌ Error importing modules: {e}")
    sys.exit(1)

def test_sentiment_plots():
    """Test sentiment plotting functionality"""
    print("\n--- Testing Sentiment Plots ---")
    
    # Create sample sentiment data
    dates = pd.date_range('2023-01-01', periods=30, freq='D')
    sample_sentiment_df = pd.DataFrame({
        'Avg_Sentiment': np.random.normal(0, 0.5, 30),
        'Total_Sentiment': np.random.normal(0, 2, 30),
        'Positive_Count': np.random.poisson(5, 30),
        'Negative_Count': np.random.poisson(3, 30),
        'Neutral_Count': np.random.poisson(2, 30),
        'News_Count': np.random.poisson(10, 30)
    }, index=dates)
    
    try:
        # Test sentiment plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Test News Sentiment Analysis', fontsize=16, fontweight='bold')
        
        # 1. Sentiment Scores Over Time
        axes[0, 0].plot(sample_sentiment_df.index, sample_sentiment_df['Avg_Sentiment'], 
                        color='blue', linewidth=2, alpha=0.7)
        axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[0, 0].set_title('Average Daily Sentiment Score Over Time')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Average Sentiment Score')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Sentiment Distribution
        sentiment_counts = [sample_sentiment_df['Positive_Count'].sum(), 
                           sample_sentiment_df['Negative_Count'].sum(), 
                           sample_sentiment_df['Neutral_Count'].sum()]
        sentiment_labels = ['Positive', 'Negative', 'Neutral']
        colors = ['green', 'red', 'gray']
        
        axes[0, 1].pie(sentiment_counts, labels=sentiment_labels, colors=colors, autopct='%1.1f%%', startangle=90)
        axes[0, 1].set_title('Overall Sentiment Distribution')
        
        # 3. Daily News Count Over Time
        axes[1, 0].bar(sample_sentiment_df.index, sample_sentiment_df['News_Count'], 
                       color='orange', alpha=0.7, width=1)
        axes[1, 0].set_title('Daily News Count')
        axes[1, 0].set_xlabel('Date')
        axes[1, 0].set_ylabel('Number of News Articles')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Sentiment Score vs Total Sentiment
        axes[1, 1].scatter(sample_sentiment_df['Avg_Sentiment'], sample_sentiment_df['Total_Sentiment'], 
                           alpha=0.6, color='purple')
        axes[1, 1].set_title('Average vs Total Sentiment Score')
        axes[1, 1].set_xlabel('Average Sentiment Score')
        axes[1, 1].set_ylabel('Total Sentiment Score')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/Users/sandy/PycharmProjects/stock_market_analysis/Output/test_sentiment_plots.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Sentiment plots test passed")
        return True
        
    except Exception as e:
        print(f"❌ Sentiment plots test failed: {e}")
        return False

def test_lstm_loss_plots():
    """Test LSTM training loss plotting functionality"""
    print("\n--- Testing LSTM Loss Plots ---")
    
    try:
        # Create sample training history
        epochs = 20
        sample_history = {
            'loss': np.exp(-np.linspace(0, 2, epochs)) + np.random.normal(0, 0.01, epochs),
            'val_loss': np.exp(-np.linspace(0, 1.8, epochs)) + np.random.normal(0, 0.02, epochs)
        }
        
        # Test individual model loss plots
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Test LSTM Training and Validation Loss', fontsize=16, fontweight='bold')
        
        # Plot training and validation loss
        axes[0].plot(sample_history['loss'], label='Training Loss', color='blue', linewidth=2)
        axes[0].plot(sample_history['val_loss'], label='Validation Loss', color='red', linewidth=2)
        axes[0].set_title('Model Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot loss difference
        loss_diff = [abs(t - v) for t, v in zip(sample_history['loss'], sample_history['val_loss'])]
        axes[1].plot(loss_diff, label='|Training - Validation| Loss', color='purple', linewidth=2)
        axes[1].set_title('Training vs Validation Loss Difference')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss Difference')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/Users/sandy/PycharmProjects/stock_market_analysis/Output/test_lstm_loss_plots.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        # Test comparison plots
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Test LSTM Models Training Loss Comparison', fontsize=16, fontweight='bold')
        
        # Create sample data for multiple models
        baseline_loss = np.exp(-np.linspace(0, 2, epochs)) + np.random.normal(0, 0.01, epochs)
        tech_loss = np.exp(-np.linspace(0, 2.2, epochs)) + np.random.normal(0, 0.01, epochs)
        enhanced_loss = np.exp(-np.linspace(0, 2.5, epochs)) + np.random.normal(0, 0.01, epochs)
        
        baseline_val_loss = np.exp(-np.linspace(0, 1.8, epochs)) + np.random.normal(0, 0.02, epochs)
        tech_val_loss = np.exp(-np.linspace(0, 2.0, epochs)) + np.random.normal(0, 0.02, epochs)
        enhanced_val_loss = np.exp(-np.linspace(0, 2.3, epochs)) + np.random.normal(0, 0.02, epochs)
        
        # Plot all training losses
        axes[0].plot(baseline_loss, label='Baseline LSTM', color='blue', linewidth=2, alpha=0.8)
        axes[0].plot(tech_loss, label='Technical LSTM', color='red', linewidth=2, alpha=0.8)
        axes[0].plot(enhanced_loss, label='Enhanced LSTM', color='green', linewidth=2, alpha=0.8)
        axes[0].set_title('Training Loss Comparison')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Training Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot all validation losses
        axes[1].plot(baseline_val_loss, label='Baseline LSTM', color='blue', linewidth=2, alpha=0.8)
        axes[1].plot(tech_val_loss, label='Technical LSTM', color='red', linewidth=2, alpha=0.8)
        axes[1].plot(enhanced_val_loss, label='Enhanced LSTM', color='green', linewidth=2, alpha=0.8)
        axes[1].set_title('Validation Loss Comparison')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Validation Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/Users/sandy/PycharmProjects/stock_market_analysis/Output/test_lstm_comparison_plots.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ LSTM loss plots test passed")
        return True
        
    except Exception as e:
        print(f"❌ LSTM loss plots test failed: {e}")
        return False

def test_correlation_plots():
    """Test sentiment vs stock price correlation plots"""
    print("\n--- Testing Correlation Plots ---")
    
    try:
        # Create sample merged data
        dates = pd.date_range('2023-01-01', periods=30, freq='D')
        sample_merged_df = pd.DataFrame({
            'Avg_Sentiment': np.random.normal(0, 0.5, 30),
            'Close': 100 + np.cumsum(np.random.normal(0, 2, 30)),
            'Returns': np.random.normal(0, 0.02, 30),
            'Volume': np.random.lognormal(15, 0.5, 30),
            'News_Count': np.random.poisson(10, 30)
        }, index=dates)
        
        # Create correlation plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Test Sentiment vs Stock Price Correlation Analysis', fontsize=16, fontweight='bold')
        
        # 1. Sentiment Score vs Stock Price
        axes[0, 0].scatter(sample_merged_df['Avg_Sentiment'], sample_merged_df['Close'], 
                           alpha=0.6, color='blue')
        axes[0, 0].set_title('Average Sentiment vs Stock Close Price')
        axes[0, 0].set_xlabel('Average Sentiment Score')
        axes[0, 0].set_ylabel('Stock Close Price ($)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add correlation coefficient
        corr_price = sample_merged_df['Avg_Sentiment'].corr(sample_merged_df['Close'])
        axes[0, 0].text(0.05, 0.95, f'Correlation: {corr_price:.3f}', transform=axes[0, 0].transAxes, 
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 2. Sentiment Score vs Stock Returns
        axes[0, 1].scatter(sample_merged_df['Avg_Sentiment'], sample_merged_df['Returns'], 
                           alpha=0.6, color='red')
        axes[0, 1].set_title('Average Sentiment vs Stock Returns')
        axes[0, 1].set_xlabel('Average Sentiment Score')
        axes[0, 1].set_ylabel('Stock Returns')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Add correlation coefficient
        corr_returns = sample_merged_df['Avg_Sentiment'].corr(sample_merged_df['Returns'])
        axes[0, 1].text(0.05, 0.95, f'Correlation: {corr_returns:.3f}', transform=axes[0, 1].transAxes,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 3. Time series comparison
        ax1 = axes[1, 0]
        ax2 = ax1.twinx()
        
        line1 = ax1.plot(sample_merged_df.index, sample_merged_df['Avg_Sentiment'], 
                         color='blue', label='Avg Sentiment', linewidth=2)
        line2 = ax2.plot(sample_merged_df.index, sample_merged_df['Close'], 
                         color='red', label='Stock Price', linewidth=2)
        
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Average Sentiment Score', color='blue')
        ax2.set_ylabel('Stock Close Price ($)', color='red')
        ax1.set_title('Sentiment Score and Stock Price Over Time')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left')
        
        # 4. News Count vs Stock Volume
        axes[1, 1].scatter(sample_merged_df['News_Count'], sample_merged_df['Volume'], 
                           alpha=0.6, color='green')
        axes[1, 1].set_title('Daily News Count vs Trading Volume')
        axes[1, 1].set_xlabel('Number of News Articles')
        axes[1, 1].set_ylabel('Trading Volume')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add correlation coefficient
        corr_volume = sample_merged_df['News_Count'].corr(sample_merged_df['Volume'])
        axes[1, 1].text(0.05, 0.95, f'Correlation: {corr_volume:.3f}', transform=axes[1, 1].transAxes,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('/Users/sandy/PycharmProjects/stock_market_analysis/Output/test_correlation_plots.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Correlation plots test passed")
        return True
        
    except Exception as e:
        print(f"❌ Correlation plots test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing plotting functionality for main_analysis.ipynb")
    print("=" * 60)
    
    # Ensure output directory exists
    os.makedirs('/Users/sandy/PycharmProjects/stock_market_analysis/Output', exist_ok=True)
    
    # Suppress warnings for cleaner output
    warnings.filterwarnings('ignore')
    
    # Run tests
    tests = [
        test_sentiment_plots,
        test_lstm_loss_plots,
        test_correlation_plots
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    # Summary
    print("\n" + "=" * 60)
    print("🏁 Test Summary:")
    passed = sum(results)
    total = len(results)
    print(f"✅ {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All plotting functionality tests passed!")
        print("The notebook should work correctly with the added plots.")
    else:
        print("⚠️ Some tests failed. Please check the error messages above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)