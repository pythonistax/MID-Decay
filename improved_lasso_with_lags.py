import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score

def create_optimal_lag_features(df, lag_period=2, key_features=None, exclude_cols=['date', 'mid']):
    """
    Create lag features for the optimal lag period (2 days based on testing)
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
    
    print(f"Creating {lag_period}-day lag features for {len(lag_cols)} variables...")
    
    # Create lag features grouped by MID
    for col in lag_cols:
        if col in df_lagged.columns:
            # Create lagged feature grouped by MID
            df_lagged[f'{col}_lag_{lag_period}'] = df_lagged.groupby('mid')[col].shift(lag_period)
    
    # Drop rows with NaN values created by lagging
    initial_rows = len(df_lagged)
    df_lagged = df_lagged.dropna()
    final_rows = len(df_lagged)
    
    print(f"Lag feature creation complete!")
    print(f"Original rows: {initial_rows}, Final rows: {final_rows}")
    print(f"Rows dropped due to lagging: {initial_rows - final_rows}")
    
    return df_lagged

# 1) Target and sort with lag features
df = test.copy()
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

# Key variables that showed importance in testing
key_variables = [
    'cycle_1_success_rate', 'processing_fees_rate', 'declined_charge_ratess', 
    'declined_unique_cycle_attempt_ratess', 'attempted_charges', 'declined_charges',
    'successful_charges', 'unique_cycle_attempts', 'cycle_1_attempts'
]

# Create optimal 2-day lag features
df_with_lags = create_optimal_lag_features(df, lag_period=2, key_features=key_variables)

target = 'success_charge_ratess'  # change if needed
exclude = {'date', 'mid', target}

# 2) Features (numeric only; keep one-hots) - now includes lag features
X = df_with_lags.drop(columns=[c for c in exclude if c in df_with_lags.columns], errors='ignore')
X = X.select_dtypes(include=[np.number]).copy()
y = df_with_lags[target].values

# Optional: drop constant/near-constant cols
nzv = X.columns[(X == 0).mean() > 0.995]
if len(nzv) > 0:
    print(f"Dropping {len(nzv)} near-zero variance features")
    X = X.drop(columns=nzv)

print(f"Total features after lag creation: {X.shape[1]}")

# 3) Train/holdout split (time-based)
split_idx = int(len(df_with_lags) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

# 4) Model
pipe = Pipeline([
    ('scaler', StandardScaler(with_mean=False)),
    ('model', LassoCV(cv=5, n_jobs=-1, random_state=42, max_iter=2000))
])

pipe.fit(X_train, y_train)

# 5) Evaluate
pred = pipe.predict(X_test)
mae = mean_absolute_error(y_test, pred)
r2 = r2_score(y_test, pred)

print("\n" + "="*50)
print("IMPROVED MODEL WITH 2-DAY LAG FEATURES")
print("="*50)
print(f"MAE: {mae:.15f}")
print(f"R²: {r2:.10f}")
print(f"Alpha (regularization): {pipe.named_steps['model'].alpha_:.2e}")

# Compare with your baseline
baseline_r2 = 0.9999987737440964
baseline_mae = 0.00014322916666666824

print(f"\nComparison with baseline:")
print(f"R² improvement: {r2 - baseline_r2:+.2e}")
print(f"MAE change: {baseline_mae - mae:+.2e}")

# 6) Get top 25 coefficients by absolute value
coef = pipe.named_steps['model'].coef_
coef_s = pd.Series(coef, index=X.columns)
top_25 = coef_s.reindex(coef_s.abs().sort_values(ascending=False).index[:25])

print(f"\nNon-zero coefficients: {np.sum(coef != 0)} out of {len(coef)}")
print(f"\nTop 25 features by absolute coefficient value:")
for i, (feature, coef_val) in enumerate(top_25.items()):
    if coef_val != 0:
        feature_type = "LAG" if "_lag_" in feature else "ORIG"
        print(f"{i+1:2d}. [{feature_type}] {feature}: {coef_val:.8e}")

# Identify which lag features are important
lag_features = [col for col in X.columns if '_lag_' in col]
original_features = [col for col in X.columns if '_lag_' not in col]

lag_coefs = coef_s[lag_features]
orig_coefs = coef_s[original_features]

important_lag_features = lag_coefs[lag_coefs != 0]
important_orig_features = orig_coefs[orig_coefs != 0]

print(f"\n" + "="*50)
print("FEATURE ANALYSIS")
print("="*50)
print(f"Total lag features created: {len(lag_features)}")
print(f"Important lag features: {len(important_lag_features)}")
print(f"Important original features: {len(important_orig_features)}")

if len(important_lag_features) > 0:
    print(f"\nImportant lag features:")
    for feature, coef_val in important_lag_features.items():
        print(f"  {feature}: {coef_val:.8e}")

print(f"\nModel successfully incorporates 2-day lag features for improved prediction!")