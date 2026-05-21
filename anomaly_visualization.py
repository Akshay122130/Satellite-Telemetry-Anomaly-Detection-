import numpy as np
import matplotlib.pyplot as plt
import os

def load_data(file_path):
    """
    Loads telemetry data from a numpy file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    return np.load(file_path)

def detect_anomalies(data, threshold=2):
    """
    Detects anomalies using the mean and standard deviation.
    Threshold defaults to 2 (mean ± 2 * std).
    """
    mean = np.mean(data)
    std = np.std(data)
    
    # Define upper and lower bounds
    upper_bound = mean + threshold * std
    lower_bound = mean - threshold * std
    
    # Find indices of anomalies
    anomalies_mask = (data > upper_bound) | (data < lower_bound)
    
    return anomalies_mask, mean, std, upper_bound, lower_bound

def plot_anomalies(data, anomalies_mask, output_path, title_suffix=""):
    """
    Plots the time-series data and highlights anomalies.
    """
    plt.figure(figsize=(15, 7))
    
    # Plot the original time-series
    plt.plot(data, label='Telemetry Data', color='blue', alpha=0.6, linewidth=1)
    
    # Highlight anomalies with red markers
    anomaly_indices = np.where(anomalies_mask)[0]
    anomaly_values = data[anomalies_mask]
    
    if len(anomaly_indices) > 0:
        plt.scatter(anomaly_indices, anomaly_values, color='red', label='Detected Anomalies', s=10)
    
    plt.title(f'Anomaly Detection in Telemetry Data {title_suffix}')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plt.savefig(output_path)
    print(f"Plot saved to: {output_path}")
    print(f"Total anomalies detected: {len(anomaly_indices)}")
    plt.close()

if __name__ == "__main__":
    # Path to the data file (using the verified path from previous analysis)
    # Using P-1.npy as an example
    data_file = os.path.join('data', 'data', 'train', 'P-1.npy')
    output_plot = os.path.join('outputs', 'plots', 'anomaly_highlight.png')
    
    try:
        # 1. Load telemetry data
        telemetry_data = load_data(data_file)
        
        # As data has 25 features, we'll demonstrate anomaly detection on the first feature
        # (Common practice for telemetry analysis is per-channel)
        feature_index = 0
        single_feature_data = telemetry_data[:, feature_index]
        
        # 2, 3. Compute stats and detect anomalies (mean ± 2 * std)
        anomalies_mask, mean, std, ub, lb = detect_anomalies(single_feature_data, threshold=2)
        
        print(f"Analysis for Feature {feature_index}:")
        print(f"Mean: {mean:.4f}, Std: {std:.4f}")
        print(f"Thresholds: [{lb:.4f}, {ub:.4f}]")
        
        # 4, 5, 6. Visualize and save plot
        plot_anomalies(single_feature_data, anomalies_mask, output_plot, title_suffix=f"(Feature {feature_index})")
        
    except Exception as e:
        print(f"Error: {e}")
