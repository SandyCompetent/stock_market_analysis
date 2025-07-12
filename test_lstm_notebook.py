# Test script to verify the LSTM notebook dependencies and basic functionality
import sys
import subprocess

def test_imports():
    """Test if all required packages are available"""
    required_packages = [
        'numpy',
        'pandas', 
        'matplotlib',
        'seaborn',
        'yfinance',
        'sklearn',
        'tensorflow',
        'scipy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} - Available")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} - Missing")
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Install missing packages with: pip install " + " ".join(missing_packages))
        return False
    else:
        print("\n✅ All required packages are available!")
        return True

def test_basic_functionality():
    """Test basic functionality of key components"""
    try:
        import yfinance as yf
        import pandas as pd
        import numpy as np
        from sklearn.preprocessing import MinMaxScaler
        
        print("\n🧪 Testing basic functionality...")
        
        # Test yfinance data fetch
        print("Testing yfinance data fetch...")
        ticker = yf.Ticker("AAPL")
        data = ticker.history(period="5d")
        if not data.empty:
            print("✅ yfinance data fetch - Working")
        else:
            print("❌ yfinance data fetch - Failed")
            return False
        
        # Test MinMaxScaler
        print("Testing MinMaxScaler...")
        scaler = MinMaxScaler()
        test_data = np.array([[1, 2], [3, 4], [5, 6]])
        scaled = scaler.fit_transform(test_data)
        if scaled.shape == test_data.shape:
            print("✅ MinMaxScaler - Working")
        else:
            print("❌ MinMaxScaler - Failed")
            return False
        
        # Test TensorFlow/Keras
        print("Testing TensorFlow/Keras...")
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense
        
        model = Sequential([
            LSTM(10, input_shape=(5, 1)),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        print("✅ TensorFlow/Keras LSTM - Working")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {str(e)}")
        return False

def main():
    print("🔍 LSTM Notebook Dependency Test")
    print("=" * 40)
    
    # Test imports
    imports_ok = test_imports()
    
    if imports_ok:
        # Test basic functionality
        functionality_ok = test_basic_functionality()
        
        if functionality_ok:
            print("\n🎉 All tests passed! The LSTM notebook should work correctly.")
            print("\n📝 Next steps:")
            print("1. Open the notebook: USAStockMarket/analysis_using_lstm.ipynb")
            print("2. Modify the STOCK_TICKER parameter to analyze different stocks")
            print("3. Adjust START_DATE and END_DATE for different time periods")
            print("4. Run all cells to perform the analysis")
        else:
            print("\n⚠️  Some functionality tests failed. Check the errors above.")
    else:
        print("\n⚠️  Missing required packages. Install them before running the notebook.")

if __name__ == "__main__":
    main()