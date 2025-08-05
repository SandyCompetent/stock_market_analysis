#!/usr/bin/env python3
"""
Test script to verify that the correlation plots still work correctly 
after fixing the 'Date' column/index ambiguity error.
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
    print("✅ Successfully imported project modules")
except ImportError as e:
    print(f"❌ Error importing modules: {e}")
    sys.exit(1)

def test_correlation_plots_with_fix():
    """Test that correlation plots work correctly with the fixed code"""
    print("\n--- Testing Correlation Plots with Fixed Code ---")
    
    try:
        # Create sample tech_data with datetime index (simulating intraday data)
        dates = pd.date_range('2023-01-01 09:30:00', periods=200, freq='30min', tz='UTC')
        tech_data = pd.DataFrame({
            'Close': 100 + np.cumsum(np.random.normal(0, 0.5, 200)),
            'Returns': np.random.normal(0, 0.02, 200),
            'Volume': np.random.lognormal(15, 0.3, 200),
            'High': 100 + np.cumsum(np.random.normal(0, 0.5, 200)) + np.random.uniform(0, 1, 200),
            'Low': 100 + np.cumsum(np.random.normal(0, 0.5, 200)) - np.random.uniform(0, 1, 200),
            'Open': 100 + np.cumsum(np.random.normal(0, 0.5, 200))
        }, index=dates)
        
        # Create sample daily sentiment data with more realistic correlations
        daily_dates = pd.date_range('2023-01-01', periods=10, freq='D')
        np.random.seed(42)  # For reproducible results
        base_sentiment = np.random.normal(0, 0.3, 10)
        daily_sentiment_df = pd.DataFrame({
            'Avg_Sentiment': base_sentiment,
            'Total_Sentiment': base_sentiment * np.random.uniform(8, 12, 10),
            'Positive_Count': np.random.poisson(8, 10),
            'Negative_Count': np.random.poisson(4, 10),
            'Neutral_Count': np.random.poisson(3, 10),
            'News_Count': np.random.poisson(15, 10)
        }, index=daily_dates)
        
        print("✅ Sample data created successfully")
        
        # Apply the FIXED correlation analysis code
        print("Applying fixed correlation analysis code...")
        
        # Prepare data for correlation analysis (FIXED VERSION)
        tech_data_daily = tech_data.copy()
        tech_data_daily.index = tech_data_daily.index.tz_localize(None) if tech_data_daily.index.tz is not None else tech_data_daily.index
        
        # Group by date directly using the index to avoid column/index ambiguity
        tech_data_daily = tech_data_daily.groupby(tech_data_daily.index.date).agg({
            'Close': 'last',
            'Returns': 'sum',
            'Volume': 'sum'
        })
        
        # Convert the index back to datetime
        tech_data_daily.index = pd.to_datetime(tech_data_daily.index)
        
        # Merge sentiment and stock data
        sentiment_stock_merged = daily_sentiment_df.join(tech_data_daily, how='inner')
        
        print(f"✅ Data preparation completed, merged data shape: {sentiment_stock_merged.shape}")
        
        if not sentiment_stock_merged.empty:
            # Create correlation plots (same as in the notebook)
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Test: Sentiment vs Stock Price Correlation Analysis', fontsize=16, fontweight='bold')
            
            # 1. Sentiment Score vs Stock Price
            axes[0, 0].scatter(sentiment_stock_merged['Avg_Sentiment'], sentiment_stock_merged['Close'], 
                               alpha=0.6, color='blue')
            axes[0, 0].set_title('Average Sentiment vs Stock Close Price')
            axes[0, 0].set_xlabel('Average Sentiment Score')
            axes[0, 0].set_ylabel('Stock Close Price ($)')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Add correlation coefficient
            corr_price = sentiment_stock_merged['Avg_Sentiment'].corr(sentiment_stock_merged['Close'])
            axes[0, 0].text(0.05, 0.95, f'Correlation: {corr_price:.3f}', transform=axes[0, 0].transAxes, 
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # 2. Sentiment Score vs Stock Returns
            axes[0, 1].scatter(sentiment_stock_merged['Avg_Sentiment'], sentiment_stock_merged['Returns'], 
                               alpha=0.6, color='red')
            axes[0, 1].set_title('Average Sentiment vs Stock Returns')
            axes[0, 1].set_xlabel('Average Sentiment Score')
            axes[0, 1].set_ylabel('Stock Returns')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Add correlation coefficient
            corr_returns = sentiment_stock_merged['Avg_Sentiment'].corr(sentiment_stock_merged['Returns'])
            axes[0, 1].text(0.05, 0.95, f'Correlation: {corr_returns:.3f}', transform=axes[0, 1].transAxes,
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # 3. Time series comparison
            ax1 = axes[1, 0]
            ax2 = ax1.twinx()
            
            line1 = ax1.plot(sentiment_stock_merged.index, sentiment_stock_merged['Avg_Sentiment'], 
                             color='blue', label='Avg Sentiment', linewidth=2)
            line2 = ax2.plot(sentiment_stock_merged.index, sentiment_stock_merged['Close'], 
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
            axes[1, 1].scatter(sentiment_stock_merged['News_Count'], sentiment_stock_merged['Volume'], 
                               alpha=0.6, color='green')
            axes[1, 1].set_title('Daily News Count vs Trading Volume')
            axes[1, 1].set_xlabel('Number of News Articles')
            axes[1, 1].set_ylabel('Trading Volume')
            axes[1, 1].grid(True, alpha=0.3)
            
            # Add correlation coefficient
            corr_volume = sentiment_stock_merged['News_Count'].corr(sentiment_stock_merged['Volume'])
            axes[1, 1].text(0.05, 0.95, f'Correlation: {corr_volume:.3f}', transform=axes[1, 1].transAxes,
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            plt.savefig('/Users/sandy/PycharmProjects/stock_market_analysis/Output/test_fixed_correlation_plots.png', 
                        dpi=300, bbox_inches='tight')
            plt.close()
            
            # Print correlation summary
            print(f"\n--- Correlation Summary ---")
            print(f"Sentiment vs Stock Price: {corr_price:.4f}")
            print(f"Sentiment vs Stock Returns: {corr_returns:.4f}")
            print(f"News Count vs Trading Volume: {corr_volume:.4f}")
            print("✅ Correlation plots created and saved successfully")
            
        else:
            print("⚠️ No overlapping dates found between sentiment and stock data for correlation analysis.")
            return False
        
        print("✅ All correlation plot operations completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Correlation plots test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run the correlation plots test"""
    print("🧪 Testing correlation plots with fixed code")
    print("=" * 60)
    
    # Ensure output directory exists
    os.makedirs('/Users/sandy/PycharmProjects/stock_market_analysis/Output', exist_ok=True)
    
    # Suppress warnings for cleaner output
    warnings.filterwarnings('ignore')
    
    # Test the correlation plots
    plots_work = test_correlation_plots_with_fix()
    
    # Summary
    print("\n" + "=" * 60)
    print("🏁 Test Summary:")
    
    if plots_work:
        print("✅ Correlation plots work correctly with the fixed code")
        print("🎉 The fix maintains all visualization functionality!")
        print("📊 Check Output/test_fixed_correlation_plots.png to see the generated plots")
    else:
        print("❌ Correlation plots have issues with the fixed code")
        print("⚠️ The fix may need further adjustment")
    
    return plots_work

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)