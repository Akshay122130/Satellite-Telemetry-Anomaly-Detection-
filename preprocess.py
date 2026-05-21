import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    """
    Loads telemetry data from a numpy file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    return np.load(file_path)

def preprocess_data(data):
    """
    Handles missing values and normalizes data.
    Returns the scaled data and the scaler object.
    """
    # 1. Handle missing values using nan_to_num
    # This replaces NaN with zero and infinity with large finite numbers
    data_cleaned = np.nan_to_num(data)
    
    # 2. Normalize data using StandardScaler (Z-score normalization)
    scaler = StandardScaler()
    # If data is 1D, we need to reshape it for the scaler
    is_1d = len(data_cleaned.shape) == 1
    if is_1d:
        data_cleaned = data_cleaned.reshape(-1, 1)
        
    scaled_data = scaler.fit_transform(data_cleaned)
    
    # If it was originally 1D, flatten it back (optional, but good for consistency)
    if is_1d:
        scaled_data = scaled_data.flatten()
        
    return scaled_data, scaler

def visualize_preprocessing(original_data, scaled_data, output_path, feature_index=0):
    """
    Plots original vs normalized data in the same figure for comparison.
    """
    # Handle multidimensional data by selecting a specific feature
    if len(original_data.shape) > 1:
        orig_plot = original_data[:, feature_index]
        scaled_plot = scaled_data[:, feature_index]
        feat_name = f" (Feature {feature_index})"
    else:
        orig_plot = original_data
        scaled_plot = scaled_data
        feat_name = ""

    plt.figure(figsize=(15, 10))
    
    # Subplot 1: Original Data
    plt.subplot(2, 1, 1)
    plt.plot(orig_plot, color='blue', alpha=0.7)
    plt.title(f'Original Telemetry Data{feat_name}')
    plt.xlabel('Time Step')
    plt.ylabel('Original Value')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Subplot 2: Scaled Data
    plt.subplot(2, 1, 2)
    plt.plot(scaled_plot, color='green', alpha=0.7)
    plt.title(f'Normalized Telemetry Data (StandardScaler){feat_name}')
    plt.xlabel('Time Step')
    plt.ylabel('Scaled Value (Z-score)')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plt.savefig(output_path)
    print(f"Preprocessing comparison plot saved to: {output_path}")
    plt.close()

if __name__ == "__main__":
    # Path to the data file
    data_file = os.path.join('data', 'data', 'train', 'P-1.npy')
    output_plot = os.path.join('outputs', 'plots', 'preprocessing_comparison.png')
    
    try:
        # 1. Load data
        raw_data = load_data(data_file)
        
        # 2, 3. Preprocess data (returns scaled data and scaler)
        # Note: preprocess_data uses nan_to_num and MinMaxScaler
        scaled_data, scaler = preprocess_data(raw_data)
        
        print("Data Preprocessing Complete.")
        print(f"Original shape: {raw_data.shape}")
        print(f"Scaled shape:   {scaled_data.shape}")
        print(f"Scaled Min:     {np.min(scaled_data):.4f}")
        print(f"Scaled Max:     {np.max(scaled_data):.4f}")
        
        # 4, 5. Visualize comparison and save plot
        visualize_preprocessing(raw_data, scaled_data, output_plot, feature_index=0)
        
    except Exception as e:
        print(f"Error during preprocessing: {e}")
