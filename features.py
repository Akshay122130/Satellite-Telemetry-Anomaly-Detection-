import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

def create_features(data, window_size=5, lag_shift=1):
    """
    Creates multiple features from preprocessed telemetry data.
    
    Includes:
    - Original data
    - Rolling mean (window size = 5)
    - Lag feature (shift by 1)
    - Rolling standard deviation (window size = 5)
    - Difference between consecutive values
    - Z-score normalization
    
    Args:
        data (np.ndarray): Preprocessed telemetry data (shape: n_samples, n_features)
        window_size (int): Window size for rolling mean and std
        lag_shift (int): Number of steps to shift for lag feature
        
    Returns:
        np.ndarray: Final feature matrix.
    """
    # Convert to DataFrame for easier time-series manipulations
    if isinstance(data, np.ndarray):
        df = pd.DataFrame(data)
    else:
        df = data.copy()
    
    original_cols = df.columns
    
    # 1. Rolling Mean
    df_rolling_mean = df.rolling(window=window_size).mean()
    df_rolling_mean.columns = [f"rolling_mean_{c}" for c in original_cols]
    
    # 2. Lag Feature
    df_lag = df.shift(periods=lag_shift)
    df_lag.columns = [f"lag_{c}" for c in original_cols]
    
    # 3. Rolling Standard Deviation
    df_rolling_std = df.rolling(window=window_size).std()
    df_rolling_std.columns = [f"rolling_std_{c}" for c in original_cols]
    
    # 4. Difference between consecutive values
    df_diff = df.diff()
    df_diff.columns = [f"diff_{c}" for c in original_cols]
    
    # Combine: Original, Rolling Mean, Lag, Rolling Std, Diff
    feature_matrix = pd.concat([df, df_rolling_mean, df_lag, df_rolling_std, df_diff], axis=1)
    
    # Ensure all column names are strings for scikit-learn
    feature_matrix.columns = feature_matrix.columns.astype(str)
    
    # Handle NaN values introduced by rolling and diff
    feature_matrix = feature_matrix.bfill()
    feature_matrix = feature_matrix.fillna(0)
    
    # 5. Z-score normalization (Standard Scaling)
    # Applying it on the final feature matrix
    scaler = StandardScaler()
    feature_matrix_scaled = scaler.fit_transform(feature_matrix)
    
    return feature_matrix_scaled

if __name__ == "__main__":
    from preprocess import load_data, preprocess_data
    
    data_file = os.path.join('data', 'data', 'train', 'P-1.npy')
    
    try:
        # 1. Load and Preprocess data
        raw_data = load_data(data_file)
        scaled_data, _ = preprocess_data(raw_data)
        
        # 2. Create features
        # For (2872, 25), result should be (2872, 125) [25 * 5]
        feature_matrix = create_features(scaled_data, window_size=5, lag_shift=1)
        
        print("Feature Engineering Complete.")
        print(f"Original shape: {scaled_data.shape}")
        print(f"Feature matrix shape: {feature_matrix.shape}")
        
    except Exception as e:
        print(f"Error during feature engineering: {e}")
