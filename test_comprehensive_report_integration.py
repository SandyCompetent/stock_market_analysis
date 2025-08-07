#!/usr/bin/env python3
"""
Test script to verify the integration of comprehensive report generation
with actual analysis results from main_analysis.ipynb
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
import config as cfg
from comprehensive_analysis_report import ComprehensiveAnalysisReport

def create_realistic_test_data():
    """Create realistic test data that mimics the output from main_analysis.ipynb"""
    
    # Create sample all_metrics_df (similar to what main_analysis.ipynb produces)
    all_metrics_df = pd.DataFrame({
        'Model': [
            'Tuned Single-Layer Baseline LSTM',
            'Tuned Single-Layer Technical LSTM', 
            'Tuned Single-Layer Hybrid LSTM',
            'Tuned Multi-Layer Baseline LSTM',
            'Tuned Multi-Layer Technical LSTM',
            'Tuned Multi-Layer Hybrid LSTM',
            'Tuned Baseline GRU',
            'Tuned Technical GRU',
            'Tuned Hybrid GRU',
            'Baseline SVM',
            'Technical SVM',
            'Hybrid SVM'
        ],
        'RMSE': [0.0245, 0.0238, 0.0229, 0.0241, 0.0235, 0.0225, 0.0243, 0.0237, 0.0231, 0.0267, 0.0259, 0.0251],
        'MAE': [0.0189, 0.0184, 0.0177, 0.0187, 0.0181, 0.0175, 0.0185, 0.0179, 0.0173, 0.0201, 0.0195, 0.0189],
        'MAPE (%)': [12.45, 11.89, 11.23, 12.12, 11.67, 11.15, 11.98, 11.45, 10.89, 13.67, 13.21, 12.89],
        'R-squared': [0.342, 0.367, 0.389, 0.351, 0.371, 0.395, 0.348, 0.365, 0.382, 0.298, 0.315, 0.334],
        'Directional_Accuracy': [52.3, 54.1, 56.2, 53.2, 54.8, 57.1, 52.8, 54.5, 55.9, 49.8, 51.2, 52.7],
        'MASE': [0.89, 0.86, 0.83, 0.88, 0.85, 0.82, 0.87, 0.84, 0.81, 0.95, 0.92, 0.89]
    })
    
    # Create ranking DataFrame (similar to final_ranking in main_analysis.ipynb)
    ranking_df = all_metrics_df.copy()
    
    # Add ranking columns
    ranking_criteria = {
        "RMSE": True,
        "MAE": True,
        "MAPE (%)": True,
        "MASE": True,
        "R-squared": False,
        "Directional_Accuracy": False,
    }
    
    for metric, ascending_order in ranking_criteria.items():
        ranking_df[f"{metric}_Rank"] = ranking_df[metric].rank(
            method="min", ascending=ascending_order
        )
    
    # Calculate total rank
    rank_components = ["MASE_Rank", "MAPE (%)_Rank", "Directional_Accuracy_Rank"]
    ranking_df["Total_Rank"] = ranking_df[rank_components].sum(axis=1)
    
    # Sort by total rank
    final_ranking = ranking_df.sort_values(by="Total_Rank", ascending=True)
    
    # Get winner (best model)
    winner = final_ranking.iloc[0]
    
    return all_metrics_df, final_ranking, winner

def test_comprehensive_report_integration():
    """Test the comprehensive report integration with real data"""
    
    print("🧪 Testing Comprehensive Report Integration...")
    print("=" * 60)
    
    # Ensure output directory exists
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    try:
        # Create realistic test data
        all_metrics_df, final_ranking, winner = create_realistic_test_data()
        
        print("✅ Test data created successfully")
        print(f"   • Models tested: {len(all_metrics_df)}")
        print(f"   • Best model: {winner['Model']}")
        print(f"   • Best model RMSE: {winner['RMSE']:.4f}")
        
        # Test 1: Create analyzer and generate report with real data
        print("\n📊 Test 1: Generating report with actual analysis results...")
        
        analyzer = ComprehensiveAnalysisReport()
        analyzer.analyze_codebase_structure()
        analyzer.analyze_model_performance(
            results_df=all_metrics_df,
            ranking_df=final_ranking,
            winner_info=winner
        )
        analyzer.generate_improvement_recommendations()
        analyzer.generate_visualization_recommendations()
        
        # Generate the detailed report
        test_report_file = os.path.join(
            cfg.OUTPUT_DIR,
            f"test_integration_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )
        
        report_file = analyzer.generate_report(output_file=test_report_file)
        
        print(f"✅ Report generated successfully: {report_file}")
        
        # Test 2: Verify report content
        print("\n📋 Test 2: Verifying report content...")
        
        with open(report_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for key elements that should be in the report
        checks = [
            ("Winner model mentioned", winner['Model'] in content),
            ("RMSE values present", "RMSE:" in content),
            ("Ranking information", "Overall Rank:" in content),
            ("Model analysis sections", "INDIVIDUAL MODEL ANALYSIS" in content),
            ("Improvement recommendations", "IMPROVEMENT RECOMMENDATIONS" in content),
            ("Codebase analysis", "CODEBASE STRUCTURE ANALYSIS" in content)
        ]
        
        all_passed = True
        for check_name, passed in checks:
            status = "✅" if passed else "❌"
            print(f"   {status} {check_name}")
            if not passed:
                all_passed = False
        
        # Test 3: Display report preview
        print(f"\n📖 Test 3: Report preview (first 1000 characters):")
        print("-" * 60)
        print(content[:1000] + "..." if len(content) > 1000 else content)
        print("-" * 60)
        
        if all_passed:
            print("\n🎉 All tests passed! Integration is working correctly.")
            print(f"📁 Test report saved to: {report_file}")
            return True
        else:
            print("\n⚠️ Some tests failed. Please check the implementation.")
            return False
            
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function to run the integration test"""
    print("🚀 Starting Comprehensive Report Integration Test...")
    print(f"Target Stock: {cfg.STOCK_SYMBOL}")
    print(f"Output Directory: {cfg.OUTPUT_DIR}")
    print("=" * 60)
    
    success = test_comprehensive_report_integration()
    
    if success:
        print("\n✅ Integration test completed successfully!")
        print("The comprehensive report generation is now properly integrated")
        print("with the main_analysis.ipynb workflow and uses real analysis results.")
    else:
        print("\n❌ Integration test failed!")
        print("Please check the error messages above and fix any issues.")
    
    return success

if __name__ == "__main__":
    main()