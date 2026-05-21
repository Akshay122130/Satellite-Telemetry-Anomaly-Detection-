import numpy as np
import os

def prepare_multi_sensor_data(file_path):
    """
    Converts a single-channel telemetry signal into a structured multi-sensor dataset.
    
    Why synthetic sensors are created:
    1. In real-world spacecraft systems, multiple sensors (temperature, voltage, etc.) 
       often track the same physical event or subsystem.
    2. Creating synthetic correlations allows us to simulate multi-modal telemetry 
       when only a single base signal is available.
    3. This structure helps in testing multi-variate anomaly detection algorithms 
       that rely on inter-sensor relationships.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at: {file_path}")
    
    # 1. Load dataset from numpy file
    raw_data = np.load(file_path)
    
    # 2. Extract base signal
    # If the data is multi-dimensional (e.g., 25 channels), we use the first channel 
    # as our base signal for synthetic generation.
    if len(raw_data.shape) > 1:
        base_signal = raw_data[:, 0]
    else:
        base_signal = raw_data
        
    # 3. Create synthetic sensor values
    # These represent physical variables correlated with the base signal
    temperature = base_signal
    voltage = base_signal * 1.2
    current = base_signal * 0.8
    battery = 100 - base_signal * 10
    
    # 4. Combine into feature matrix X (column stack)
    # Resulting shape: (n_samples, 4)
    X = np.column_stack((temperature, voltage, current, battery))
    
    # 5. Normalize battery values between 0 and 100
    # Battery should represent a percentage (0-100%)
    # We clip values to ensure physical constraints
    X[:, 3] = np.clip(X[:, 3], 0, 100)
    
    return X

if __name__ == "__main__":
    # Define data path
    data_file = os.path.join('data', 'data', 'train', 'P-1.npy')
    
    try:
        # Generate multi-sensor data
        X_multi = prepare_multi_sensor_data(data_file)
        
        print("--- Multi-Sensor Data Preparation ---")
        print(f"Base file: {data_file}")
        print(f"Output Feature Matrix Shape: {X_multi.shape}")
        
        # Display sample values for each synthetic sensor
        sensors = ["Temperature", "Voltage", "Current", "Battery"]
        print("\nFirst 5 samples:")
        print(f"{'Index':<8} | {'Temp':<12} | {'Volt':<12} | {'Curr':<12} | {'Batt (%)':<12}")
        print("-" * 65)
        for i in range(5):
            print(f"{i:<8} | {X_multi[i, 0]:<12.4f} | {X_multi[i, 1]:<12.4f} | {X_multi[i, 2]:<12.4f} | {X_multi[i, 3]:<12.4f}")
            
        print("\nData preparation successful.")
        
    except Exception as e:
        print(f"Error during data preparation: {e}")
