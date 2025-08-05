#!/usr/bin/env python3
"""
Test script to verify the fix for the 'Date' column/index ambiguity error
in the sentiment vs stock price correlation analysis section.
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

def test_correlation_fix():
    """Test the fixed correlation analysis code"""
    print("\n--- Testing Fixed Correlation Analysis Code ---")
    
    try:
        # Create sample tech_data with datetime index (simulating intraday data)
        dates = pd.date_range('2023-01-01 09:30:00', periods=100, freq='1H', tz='UTC')
        tech_data = pd.DataFrame({
            'Close': 100 + np.cumsum(np.random.normal(0, 1, 100)),
            'Returns': np.random.normal(0, 0.02, 100),
            'Volume': np.random.lognormal(15, 0.5, 100),
            'High': 100 + np.cumsum(np.random.normal(0, 1, 100)) + np.random.uniform(0, 2, 100),
            'Low': 100 + np.cumsum(np.random.normal(0, 1, 100)) - np.random.uniform(0, 2, 100),
            'Open': 100 + np.cumsum(np.random.normal(0, 1, 100))
        }, index=dates)
        
        # Create sample daily sentiment data
        daily_dates = pd.date_range('2023-01-01', periods=5, freq='D')
        daily_sentiment_df = pd.DataFrame({
            'Avg_Sentiment': np.random.normal(0, 0.5, 5),
            'Total_Sentiment': np.random.normal(0, 2, 5),
            'Positive_Count': np.random.poisson(5, 5),
            'Negative_Count': np.random.poisson(3, 5),
            'Neutral_Count': np.random.poisson(2, 5),
            'News_Count': np.random.poisson(10, 5)
        }, index=daily_dates)
        
        print("✅ Sample data created successfully")
        
        # Test the fixed correlation analysis code
        print("Testing the fixed groupby operation...")
        
        # Prepare data for correlation analysis (FIXED VERSION)
        # Convert tech_data index to date for merging with sentiment data
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
        
        print("✅ Groupby operation completed without error")
        print(f"✅ tech_data_daily shape: {tech_data_daily.shape}")
        print(f"✅ tech_data_daily index type: {type(tech_data_daily.index)}")
        
        # Test the merge operation
        sentiment_stock_merged = daily_sentiment_df.join(tech_data_daily, how='inner')
        print(f"✅ Merge completed successfully, merged data shape: {sentiment_stock_merged.shape}")
        
        if not sentiment_stock_merged.empty:
            print("✅ Merged data is not empty, correlation analysis can proceed")
            
            # Test correlation calculations
            if 'Close' in sentiment_stock_merged.columns and 'Avg_Sentiment' in sentiment_stock_merged.columns:
                corr_price = sentiment_stock_merged['Avg_Sentiment'].corr(sentiment_stock_merged['Close'])
                print(f"✅ Price correlation calculated: {corr_price:.4f}")
            
            if 'Returns' in sentiment_stock_merged.columns:
                corr_returns = sentiment_stock_merged['Avg_Sentiment'].corr(sentiment_stock_merged['Returns'])
                print(f"✅ Returns correlation calculated: {corr_returns:.4f}")
            
            if 'Volume' in sentiment_stock_merged.columns:
                corr_volume = sentiment_stock_merged['News_Count'].corr(sentiment_stock_merged['Volume'])
                print(f"✅ Volume correlation calculated: {corr_volume:.4f}")
                
        else:
            print("⚠️ Merged data is empty, but no error occurred")
        
        print("✅ All correlation analysis operations completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Correlation analysis test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_old_problematic_code():
    """Test the old problematic code to confirm it would fail"""
    print("\n--- Testing Old Problematic Code (Should Fail) ---")
    
    try:
        # Create sample tech_data with datetime index
        dates = pd.date_range('2023-01-01 09:30:00', periods=100, freq='1H', tz='UTC')
        tech_data = pd.DataFrame({
            'Close': 100 + np.cumsum(np.random.normal(0, 1, 100)),
            'Returns': np.random.normal(0, 0.02, 100),
            'Volume': np.random.lognormal(15, 0.5, 100)
        }, index=dates)
        
        # OLD PROBLEMATIC CODE
        tech_data_daily = tech_data.copy()
        tech_data_daily.index = tech_data_daily.index.tz_localize(None) if tech_data_daily.index.tz is not None else tech_data_daily.index
        tech_data_daily['Date'] = tech_data_daily.index.date  # This creates the ambiguity
        tech_data_daily = tech_data_daily.groupby('Date').agg({  # This should fail
            'Close': 'last',
            'Returns': 'sum',
            'Volume': 'sum'
        }).reset_index()
        
        print("⚠️ Old code unexpectedly succeeded (pandas version may handle this differently)")
        return False
        
    except ValueError as e:
        if "ambiguous" in str(e).lower():
            print(f"✅ Old code correctly failed with expected error: {e}")
            return True
        else:
            print(f"❌ Old code failed with unexpected error: {e}")
            return False
    except Exception as e:
        print(f"❌ Old code failed with unexpected error type: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing fix for 'Date' column/index ambiguity error")
    print("=" * 60)
    
    # Suppress warnings for cleaner output
    warnings.filterwarnings('ignore')
    
    # Test the fix
    fix_works = test_correlation_fix()
    
    # Test the old problematic code
    old_code_fails = test_old_problematic_code()
    
    # Summary
    print("\n" + "=" * 60)
    print("🏁 Test Summary:")
    
    if fix_works:
        print("✅ Fixed correlation analysis code works correctly")
    else:
        print("❌ Fixed correlation analysis code still has issues")
    
    if old_code_fails:
        print("✅ Old problematic code correctly fails as expected")
    else:
        print("⚠️ Old problematic code behavior differs from expected")
    
    if fix_works:
        print("🎉 The fix should resolve the ValueError in the notebook!")
        print("The correlation analysis section should now work without the ambiguity error.")
    else:
        print("⚠️ The fix may need further adjustment.")
    
    return fix_works

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)