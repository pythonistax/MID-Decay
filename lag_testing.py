"""
Lag Feature Testing for Lasso Regression
This script tests different lag periods to improve the baseline model performance.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

def create_lag_features(df, lag_periods=[1, 2, 3, 5, 7, 10, 14, 21, 30], 
                       key_features=None, exclude_cols=['date', 'mid']):
    """
    Create comprehensive lag features for time series analysis
    """
    df_lagged = df.copy()
    df_lagged['date'] = pd.to_datetime(df_lagged['date'])
    df_lagged = df_lagged.sort_values(['mid', 'date']).reset_index(drop=True)
    
    # Get numeric columns for lag creation
    if key_features is None:
        numeric_cols = df_lagged.select_dtypes(include=[np.number]).columns.tolist()
        lag_cols = [col for col in numeric_cols if col not in exclude_cols]
    else:
        lag_cols = key_features
    
    print(f"Creating lag features for {len(lag_cols)} variables across {len(lag_periods)} lag periods...")
    
    # Create lag features grouped by MID
    for lag in lag_periods:
        print(f"Processing lag period: {lag} days...")
        for col in lag_cols:
            if col in df_lagged.columns:
                # Create lagged feature grouped by MID
                df_lagged[f'{col}_lag_{lag}'] = df_lagged.groupby('mid')[col].shift(lag)
    
    # Drop rows with NaN values created by lagging
    initial_rows = len(df_lagged)
    df_lagged = df_lagged.dropna()
    final_rows = len(df_lagged)
    
    print(f"Lag feature creation complete!")
    print(f"Original rows: {initial_rows}, Final rows: {final_rows}")
    print(f"Rows dropped due to lagging: {initial_rows - final_rows}")
    
    return df_lagged

def test_lasso_with_lags(df, target='success_charge_ratess', lag_periods=[1, 2, 3, 5, 7], 
                        key_features=None, test_size=0.2):
    """
    Test Lasso regression with different lag configurations
    """
    
    # Create lag features
    df_with_lags = create_lag_features(df, lag_periods=lag_periods, key_features=key_features)
    
    # Prepare features and target
    exclude = {'date', 'mid', target}
    X = df_with_lags.drop(columns=[c for c in exclude if c in df_with_lags.columns], errors='ignore')
    X = X.select_dtypes(include=[np.number]).copy()
    y = df_with_lags[target].values
    
    # Remove near-zero variance features
    nzv = X.columns[(X == 0).mean() > 0.995]
    if len(nzv) > 0:
        print(f"Dropping {len(nzv)} near-zero variance features")
        X = X.drop(columns=nzv)
    
    # Time-based train/test split
    split_idx = int(len(df_with_lags) * (1 - test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
    
    # Fit Lasso model
    pipe = Pipeline([
        ('scaler', StandardScaler(with_mean=False)),
        ('model', LassoCV(cv=5, n_jobs=-1, random_state=42, max_iter=2000))
    ])
    
    pipe.fit(X_train, y_train)
    
    # Evaluate
    y_pred = pipe.predict(X_test)
    
    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    # Get feature importance (coefficients)
    coef = pipe.named_steps['model'].coef_
    coef_series = pd.Series(coef, index=X.columns)
    
    # Get top features by absolute coefficient value
    top_features = coef_series.reindex(coef_series.abs().sort_values(ascending=False).index)
    
    results = {
        'lag_periods': lag_periods,
        'r2': r2,
        'mae': mae,
        'rmse': rmse,
        'alpha': pipe.named_steps['model'].alpha_,
        'n_features': len(X.columns),
        'n_nonzero_coef': np.sum(coef != 0),
        'top_features': top_features.head(15),
        'feature_names': X.columns.tolist()
    }
    
    return results, pipe

def run_lag_experiments(test_data):
    """
    Run comprehensive lag experiments
    """
    # Current baseline performance for comparison
    baseline_r2 = 0.9999987737440964
    baseline_mae = 0.00014322916666666824
    
    print("=== BASELINE PERFORMANCE ===")
    print(f"R²: {baseline_r2:.10f}")
    print(f"MAE: {baseline_mae:.15f}")
    print("="*50)
    
    # Test individual lag periods
    individual_lag_results = {}
    individual_lag_periods = [1, 2, 3, 5, 7, 10, 14, 21, 30]
    
    print("\nPHASE 1: Testing Individual Lag Periods")
    print("="*60)
    
    # Focus on key variables identified from the baseline model
    key_variables = [
        'cycle_1_success_rate', 'processing_fees_rate', 'declined_charge_ratess', 
        'declined_unique_cycle_attempt_ratess', 'attempted_charges', 'declined_charges',
        'successful_charges', 'unique_cycle_attempts', 'cycle_1_attempts'
    ]
    
    for lag_period in individual_lag_periods:
        print(f"\n--- Testing Lag Period: {lag_period} days ---")
        
        try:
            results, model = test_lasso_with_lags(
                test_data, 
                target='success_charge_ratess',
                lag_periods=[lag_period],
                key_features=key_variables
            )
            
            individual_lag_results[lag_period] = results
            
            print(f"R²: {results['r2']:.10f}")
            print(f"MAE: {results['mae']:.15f}")
            print(f"RMSE: {results['rmse']:.15f}")
            print(f"Number of features: {results['n_features']}")
            print(f"Non-zero coefficients: {results['n_nonzero_coef']}")
            
            # Compare with baseline
            r2_improvement = results['r2'] - baseline_r2
            mae_improvement = baseline_mae - results['mae']  # Lower MAE is better
            
            print(f"R² vs Baseline: {r2_improvement:+.10f}")
            print(f"MAE vs Baseline: {mae_improvement:+.15f}")
            
            print("\nTop 5 Features:")
            for i, (feature, coef) in enumerate(results['top_features'].head(5).items()):
                print(f"  {i+1}. {feature}: {coef:.8e}")
                
        except Exception as e:
            print(f"Error testing lag period {lag_period}: {str(e)}")
    
    print("\n" + "="*60)
    print("PHASE 1 COMPLETE - Individual Lag Period Testing")
    
    # Find best individual lags and test combinations
    if individual_lag_results:
        # Sort by R² score 
        sorted_results = sorted(individual_lag_results.items(), 
                              key=lambda x: x[1]['r2'], reverse=True)
        
        print("\n\nINDIVIDUAL LAG PERIOD SUMMARY:")
        print("="*50)
        print("Rank | Lag | R² Score | MAE | R² vs Baseline")
        print("-" * 50)
        
        best_individual_lags = []
        for i, (lag_period, results) in enumerate(sorted_results):
            r2_diff = results['r2'] - baseline_r2
            print(f"{i+1:4d} | {lag_period:3d} | {results['r2']:.8f} | {results['mae']:.2e} | {r2_diff:+.2e}")
            
            # Select top 5 for combinations
            if i < 5:
                best_individual_lags.append(lag_period)
        
        print(f"\nSelected for combination testing: {best_individual_lags[:5]}")
        
        # Test combinations
        combination_results = {}
        lag_combinations = [
            best_individual_lags[:2],   # Top 2
            best_individual_lags[:3],   # Top 3  
            best_individual_lags[:4],   # Top 4
            [1, 7, 14],                 # Short, medium, long term
            [1, 3, 7, 21],              # Weekly progression
            [2, 5, 10, 30],             # Alternative progression
        ]
        
        print("\n\nPHASE 2: Testing Lag Combinations")
        print("="*60)
        
        for i, lag_combo in enumerate(lag_combinations):
            if not lag_combo:
                continue
                
            print(f"\n--- Combination {i+1}: {lag_combo} ---")
            
            try:
                results, model = test_lasso_with_lags(
                    test_data, 
                    target='success_charge_ratess',
                    lag_periods=lag_combo,
                    key_features=key_variables
                )
                
                combination_results[f"combo_{i+1}"] = {
                    'lag_periods': lag_combo,
                    'results': results
                }
                
                print(f"R²: {results['r2']:.10f}")
                print(f"MAE: {results['mae']:.15f}")
                print(f"RMSE: {results['rmse']:.15f}")
                print(f"Number of features: {results['n_features']}")
                print(f"Non-zero coefficients: {results['n_nonzero_coef']}")
                
                # Compare with baseline
                r2_improvement = results['r2'] - baseline_r2
                mae_improvement = baseline_mae - results['mae']
                
                print(f"R² vs Baseline: {r2_improvement:+.10f}")
                print(f"MAE vs Baseline: {mae_improvement:+.15f}")
                
                print("\nTop 5 Features:")
                for j, (feature, coef) in enumerate(results['top_features'].head(5).items()):
                    print(f"  {j+1}. {feature}: {coef:.8e}")
                    
            except Exception as e:
                print(f"Error testing combination {lag_combo}: {str(e)}")
        
        print("\n" + "="*60)
        print("PHASE 2 COMPLETE - Combination Testing")
        
        # Final analysis
        print("\n\nFINAL ANALYSIS: Best Performing Models")
        print("="*60)
        
        all_results = {}
        
        # Add individual lag results
        for lag_period, results in individual_lag_results.items():
            all_results[f"Individual_Lag_{lag_period}"] = {
                'config': f"Lag {lag_period} days",
                'lag_periods': [lag_period],
                'r2': results['r2'],
                'mae': results['mae'],
                'rmse': results['rmse']
            }
        
        # Add combination results
        for combo_name, combo_data in combination_results.items():
            results = combo_data['results']
            all_results[combo_name] = {
                'config': f"Lag combination {combo_data['lag_periods']}",
                'lag_periods': combo_data['lag_periods'],
                'r2': results['r2'],
                'mae': results['mae'],
                'rmse': results['rmse']
            }
        
        # Sort by R² score
        best_models = sorted(all_results.items(), key=lambda x: x[1]['r2'], reverse=True)
        
        print("TOP 10 MODELS BY R² SCORE:")
        print("-" * 80)
        print(f"{'Rank':<4} | {'Model':<25} | {'R² Score':<12} | {'MAE':<12} | {'R² vs Baseline':<15}")
        print("-" * 80)
        
        for i, (model_name, results) in enumerate(best_models[:10]):
            r2_diff = results['r2'] - baseline_r2
            print(f"{i+1:<4} | {model_name:<25} | {results['r2']:<12.8f} | {results['mae']:<12.2e} | {r2_diff:<+15.2e}")
        
        # Check if any model beats the baseline
        best_model = best_models[0]
        best_r2 = best_model[1]['r2']
        best_mae = best_model[1]['mae']
        
        print(f"\n{'='*60}")
        print("IMPROVEMENT ANALYSIS:")
        print(f"{'='*60}")
        print(f"Baseline Performance:")
        print(f"  R²:  {baseline_r2:.10f}")
        print(f"  MAE: {baseline_mae:.15f}")
        print()
        print(f"Best Model Performance ({best_model[0]}):")
        print(f"  R²:  {best_r2:.10f}")
        print(f"  MAE: {best_mae:.15f}")
        print()
        print(f"Improvement:")
        print(f"  R² difference:  {best_r2 - baseline_r2:+.2e}")
        print(f"  MAE difference: {baseline_mae - best_mae:+.2e}")
        
        if best_r2 > baseline_r2:
            print(f"\n🎉 SUCCESS! Best model beats baseline by {best_r2 - baseline_r2:.2e} R² points")
        elif abs(best_r2 - baseline_r2) < 1e-10:
            print(f"\n📊 EQUIVALENT: Best model performs similarly to baseline (difference: {best_r2 - baseline_r2:.2e})")
        else:
            print(f"\n📉 Baseline still superior: No improvement found with lag features")
        
        print(f"\nBest lag configuration: {best_model[1]['lag_periods']}")

if __name__ == "__main__":
    print("Lag Testing Script Ready!")
    print("To run experiments, call: run_lag_experiments(your_test_dataframe)")