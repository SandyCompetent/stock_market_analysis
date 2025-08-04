#!/usr/bin/env python3
"""
Enhanced Analysis Demo Script
============================

This script demonstrates the enhanced visualization capabilities and 
comprehensive analysis features added to the stock market prediction project.

It shows how to use the new diagnostic plots, model comparison visualizations,
and comprehensive reporting features.

Author: AI Assistant
Date: 2025-08-02
"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime

# Import project modules
import config as cfg
import utils as ut
from comprehensive_analysis_report import ComprehensiveAnalysisReport

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def create_sample_data():
    """Create sample data for demonstration purposes."""
    print("📊 Creating sample data for demonstration...")

    # Create sample dates
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    n_samples = len(dates)

    # Create sample actual values (simulating daily returns)
    np.random.seed(42)
    actual_values = np.random.normal(0.001, 0.02, n_samples)  # Mean return 0.1%, std 2%

    # Create sample predictions with some correlation to actual values
    predictions_lstm = actual_values + np.random.normal(0, 0.01, n_samples)
    predictions_gru = actual_values + np.random.normal(0, 0.012, n_samples)
    predictions_svm = actual_values + np.random.normal(0, 0.015, n_samples)
    predictions_arima = actual_values + np.random.normal(0, 0.018, n_samples)

    return {
        'dates': dates,
        'actual': actual_values,
        'lstm_pred': predictions_lstm,
        'gru_pred': predictions_gru,
        'svm_pred': predictions_svm,
        'arima_pred': predictions_arima
    }


def create_sample_results_dataframe():
    """Create a sample results dataframe with realistic performance metrics."""
    print("📈 Creating sample model performance results...")

    results_data = {
        'Model': [
            'Baseline Single-Layer LSTM',
            'Technical Multi-Layer LSTM',
            'Hybrid Enhanced LSTM',
            'Baseline GRU',
            'Technical GRU',
            'Hybrid Enhanced GRU',
            'Baseline SVM',
            'Technical SVM',
            'Hybrid Enhanced SVM',
            'ARIMA'
        ],
        'RMSE': [0.0245, 0.0238, 0.0229, 0.0241, 0.0235, 0.0231, 0.0267, 0.0259, 0.0251, 0.0289],
        'MAE': [0.0189, 0.0184, 0.0177, 0.0187, 0.0181, 0.0179, 0.0201, 0.0195, 0.0189, 0.0218],
        'MAPE (%)': [12.45, 11.89, 11.23, 12.12, 11.67, 11.45, 13.67, 13.21, 12.89, 14.23],
        'R-squared': [0.342, 0.367, 0.389, 0.351, 0.371, 0.378, 0.298, 0.315, 0.334, 0.245],
        'Directional_Accuracy': [52.3, 54.1, 56.2, 53.2, 54.8, 55.1, 49.8, 51.2, 52.7, 47.6],
        'MASE': [0.89, 0.86, 0.83, 0.88, 0.85, 0.84, 0.95, 0.92, 0.89, 1.02]
    }

    return pd.DataFrame(results_data)


def demonstrate_enhanced_diagnostics():
    """Demonstrate the enhanced diagnostic plotting capabilities."""
    print("\n🔍 Demonstrating Enhanced Diagnostic Plots...")

    # Create sample data
    sample_data = create_sample_data()

    # Demonstrate enhanced diagnostics for LSTM model
    print("Creating enhanced diagnostics for LSTM model...")
    ut.plot_enhanced_diagnostics(
        y_test=sample_data['actual'][:100],  # Use first 100 days
        predictions=sample_data['lstm_pred'][:100],
        test_dates=sample_data['dates'][:100],
        stock_symbol=cfg.STOCK_SYMBOL,
        model_name="Enhanced LSTM"
    )

    # Demonstrate enhanced diagnostics for GRU model
    print("Creating enhanced diagnostics for GRU model...")
    ut.plot_enhanced_diagnostics(
        y_test=sample_data['actual'][:100],
        predictions=sample_data['gru_pred'][:100],
        test_dates=sample_data['dates'][:100],
        stock_symbol=cfg.STOCK_SYMBOL,
        model_name="Enhanced GRU"
    )


def demonstrate_model_comparison():
    """Demonstrate the model comparison visualization capabilities."""
    print("\n📊 Demonstrating Model Comparison Visualizations...")

    # Create sample results
    results_df = create_sample_results_dataframe()

    # Create model comparison metrics plot
    print("Creating model comparison metrics plot...")
    ut.plot_model_comparison_metrics(results_df, cfg.STOCK_SYMBOL)

    return results_df


def demonstrate_top_models_comparison():
    """Demonstrate the top models comparison plot."""
    print("\n🏆 Demonstrating Top Models Comparison Plot...")

    # Create sample data
    sample_data = create_sample_data()

    # Create results dictionary for top models comparison
    plot_data = {
        "Actual": {
            "dates": sample_data['dates'][:100],
            "values": sample_data['actual'][:100]
        },
        "Hybrid Enhanced LSTM": {
            "dates": sample_data['dates'][:100],
            "values": sample_data['lstm_pred'][:100]
        },
        "Technical GRU": {
            "dates": sample_data['dates'][:100],
            "values": sample_data['gru_pred'][:100]
        },
        "Hybrid Enhanced SVM": {
            "dates": sample_data['dates'][:100],
            "values": sample_data['svm_pred'][:100]
        },
        "ARIMA": {
            "dates": sample_data['dates'][:100],
            "values": sample_data['arima_pred'][:100]
        }
    }

    # Create top models comparison plot
    print("Creating top models comparison plot...")
    ut.plot_top_models_comparison(plot_data, cfg.STOCK_SYMBOL, top_n=4)


def demonstrate_comprehensive_report():
    """Demonstrate the comprehensive analysis report generation."""
    print("\n📋 Demonstrating Comprehensive Analysis Report...")

    # Create the analyzer
    analyzer = ComprehensiveAnalysisReport()

    # Create sample results for analysis
    results_df = create_sample_results_dataframe()

    # Generate the report with actual data
    analyzer.analyze_codebase_structure()
    analyzer.analyze_model_performance(results_df)
    analyzer.generate_improvement_recommendations()
    analyzer.generate_visualization_recommendations()

    # Generate the complete report
    report_file = analyzer.generate_report()

    print(f"✅ Comprehensive report generated: {report_file}")

    # Display a preview of the report
    print("\n📖 Report Preview (first 2000 characters):")
    print("-" * 80)
    with open(report_file, 'r', encoding='utf-8') as f:
        content = f.read()
        print(content[:2000] + "..." if len(content) > 2000 else content)
    print("-" * 80)

    return report_file


def create_summary_visualization():
    """Create a summary visualization showing all enhanced features."""
    print("\n🎨 Creating Summary Visualization...")

    # Create a figure with multiple subplots showing the enhanced features
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(f'{cfg.STOCK_SYMBOL} - Enhanced Analysis Summary', fontsize=20, fontweight='bold')

    # Create sample data
    sample_data = create_sample_data()
    results_df = create_sample_results_dataframe()

    # Subplot 1: Model Performance Comparison
    ax1 = plt.subplot(2, 3, 1)
    models = results_df['Model'][:5]  # Top 5 models
    rmse_values = results_df['RMSE'][:5]
    bars = ax1.bar(range(len(models)), rmse_values, color='skyblue', alpha=0.8)
    ax1.set_title('Model RMSE Comparison', fontweight='bold')
    ax1.set_xticks(range(len(models)))
    ax1.set_xticklabels([m.split()[0] for m in models], rotation=45)
    ax1.set_ylabel('RMSE')

    # Subplot 2: Actual vs Predicted Scatter
    ax2 = plt.subplot(2, 3, 2)
    actual_sample = sample_data['actual'][:50]
    pred_sample = sample_data['lstm_pred'][:50]
    ax2.scatter(actual_sample, pred_sample, alpha=0.6, color='green')
    min_val, max_val = min(min(actual_sample), min(pred_sample)), max(max(actual_sample), max(pred_sample))
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    ax2.set_title('Actual vs Predicted', fontweight='bold')
    ax2.set_xlabel('Actual')
    ax2.set_ylabel('Predicted')

    # Subplot 3: Residuals Distribution
    ax3 = plt.subplot(2, 3, 3)
    residuals = actual_sample - pred_sample
    ax3.hist(residuals, bins=15, alpha=0.7, color='orange', edgecolor='black')
    ax3.set_title('Residuals Distribution', fontweight='bold')
    ax3.set_xlabel('Residuals')
    ax3.set_ylabel('Frequency')

    # Subplot 4: Time Series Comparison
    ax4 = plt.subplot(2, 3, 4)
    dates_sample = sample_data['dates'][:30]
    ax4.plot(dates_sample, sample_data['actual'][:30], label='Actual', color='black', linewidth=2)
    ax4.plot(dates_sample, sample_data['lstm_pred'][:30], label='LSTM', color='blue', alpha=0.8)
    ax4.plot(dates_sample, sample_data['gru_pred'][:30], label='GRU', color='red', alpha=0.8)
    ax4.set_title('Time Series Predictions', fontweight='bold')
    ax4.legend()
    ax4.tick_params(axis='x', rotation=45)

    # Subplot 5: R-squared Comparison
    ax5 = plt.subplot(2, 3, 5)
    r2_values = results_df['R-squared'][:5]
    bars = ax5.bar(range(len(models)), r2_values, color='lightcoral', alpha=0.8)
    ax5.set_title('Model R-squared Comparison', fontweight='bold')
    ax5.set_xticks(range(len(models)))
    ax5.set_xticklabels([m.split()[0] for m in models], rotation=45)
    ax5.set_ylabel('R-squared')

    # Subplot 6: Feature Importance (simulated)
    ax6 = plt.subplot(2, 3, 6)
    features = ['Close', 'Volume', 'RSI', 'MACD', 'Sentiment']
    importance = [0.35, 0.15, 0.20, 0.18, 0.12]
    bars = ax6.barh(features, importance, color='lightgreen', alpha=0.8)
    ax6.set_title('Feature Importance (Simulated)', fontweight='bold')
    ax6.set_xlabel('Importance')

    plt.tight_layout()

    # Save the summary plot
    save_path = os.path.join(cfg.OUTPUT_DIR, f"{cfg.STOCK_SYMBOL}_enhanced_analysis_summary.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Summary visualization saved to {save_path}")

    plt.show()


def main():
    """Main function to demonstrate all enhanced features."""
    print("🚀 Starting Enhanced Analysis Demonstration...")
    print(f"Target Stock: {cfg.STOCK_SYMBOL}")
    print(f"Output Directory: {cfg.OUTPUT_DIR}")
    print("=" * 80)

    start_time = time.time()

    # Ensure output directory exists
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    try:
        # Demonstrate enhanced diagnostic plots
        demonstrate_enhanced_diagnostics()

        # Demonstrate model comparison visualizations
        results_df = demonstrate_model_comparison()

        # Demonstrate top models comparison
        demonstrate_top_models_comparison()

        # Generate comprehensive analysis report
        report_file = demonstrate_comprehensive_report()

        # Create summary visualization
        create_summary_visualization()

        # Calculate and display runtime
        end_time = time.time()
        runtime = end_time - start_time

        print("\n" + "=" * 80)
        print("✅ ENHANCED ANALYSIS DEMONSTRATION COMPLETE!")
        print("=" * 80)
        print(f"⏱️  Total Runtime: {ut.format_runtime(runtime)}")
        print(f"📊 Results saved to: {cfg.OUTPUT_DIR}")
        print(f"📋 Comprehensive report: {report_file}")
        print("\n🎯 Key Enhancements Demonstrated:")
        print("   ✓ Enhanced diagnostic plots with statistical tests")
        print("   ✓ Comprehensive model comparison visualizations")
        print("   ✓ Top models performance comparison")
        print("   ✓ Detailed analysis report with recommendations")
        print("   ✓ Professional summary visualizations")
        print("\n🚀 Ready for production deployment!")

    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 All demonstrations completed successfully!")
    else:
        print("\n💥 Some demonstrations failed. Check the error messages above.")
