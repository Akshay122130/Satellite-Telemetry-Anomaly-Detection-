import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def load_telemetry_data(file_path):
    """
    Loads a numpy file containing telemetry data.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    data = np.load(file_path)
    return data

def analyze_data(data):
    """
    Performs basic analysis on the telemetry data.
    """
    print("--- Telemetry Data Analysis ---")
    print(f"Shape of data: {data.shape}")
    print(f"First 10 values:\n{data[:10]}")
    
    print("\nDescriptive Statistics:")
    print(f"Mean:   {np.mean(data):.4f}")
    print(f"StdDev: {np.std(data):.4f}")
    print(f"Min:    {np.min(data):.4f}")
    print(f"Max:    {np.max(data):.4f}")
    
    # Convert to pandas DataFrame
    df = pd.DataFrame(data, columns=[f'feature_{i}' for i in range(data.shape[1])])
    
    # Check for missing values
    missing_values = df.isnull().sum().sum()
    print(f"\nMissing values: {missing_values}")
    
    return df

def visualize_data(df, output_path):
    """
    Plots the time-series data and saves it to a file.
    """
    plt.figure(figsize=(15, 8))
    # Plotting first 5 features for clarity, or all if preferred
    num_features_to_plot = min(5, len(df.columns))
    for i in range(num_features_to_plot):
        plt.plot(df.iloc[:, i], label=df.columns[i], alpha=0.7)
    
    plt.title(f'Raw Telemetry Time-Series Data (First {num_features_to_plot} features)')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plt.savefig(output_path)
    print(f"\nPlot saved to: {output_path}")
    plt.close()

if __name__ == "__main__":
    # Path to the data file
    data_file = os.path.join('data', 'data', 'train', 'P-1.npy')
    output_plot = os.path.join('outputs', 'plots', 'raw_telemetry.png')
    
    try:
        # 1. Load data
        telemetry_data = load_telemetry_data(data_file)
        
        # 2, 3, 4. Analyze and convert to DataFrame
        df = analyze_data(telemetry_data)
        
        # 5, 6. Visualize and save plot
        visualize_data(df, output_plot)
        
    except Exception as e:
        print(f"Error: {e}")
