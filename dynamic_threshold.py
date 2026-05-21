import numpy as np

def compute_adaptive_threshold(probabilities, fallback_threshold=0.5, safety_threshold=0.1):
    """
    Computes a dynamic threshold based on the distribution of prediction probabilities.
    
    Adaptive behavior:
    1. Adjusts the percentile based on data size to maintain sensitivity:
       - Large batches (>50): 85th percentile (conservative)
       - Medium batches (>20): 80th percentile (balanced)
       - Small batches: 75th percentile (aggressive)
    2. Clamps the threshold between 0.2 and 0.8 to prevent extreme sensitivity or conservatism.
    3. Falls back to 0.5 for very small inputs (<5) where statistics are unreliable.
    4. Applies a safety threshold of 0.1 if all probabilities are extremely low.
    
    Args:
        probabilities (np.ndarray): Array of prediction probabilities (0.0 to 1.0).
        fallback_threshold (float): Threshold for very small batches (default: 0.5).
        safety_threshold (float): Minimum threshold for extremely clean data (default: 0.1).
        
    Returns:
        float: The computed dynamic threshold.
        np.ndarray: Binary predictions (1 for Anomaly, 0 for Normal).
        np.ndarray: The original probabilities.
    """
    probs = np.array(probabilities)
    data_size = len(probs)
    
    # 1. Determine adaptive percentile based on data size
    if data_size > 50:
        percentile = 85 # Conservative for large datasets
    elif data_size > 20:
        percentile = 80 # Standard balanced approach
    else:
        percentile = 75 # Aggressive for smaller samples
        
    # 2 & 4. Compute threshold using percentile with fallback logic
    if data_size < 5:
        threshold = fallback_threshold
    else:
        # Compute threshold using adaptive percentile
        threshold = np.percentile(probs, percentile)
        
        # 5. Safety logic: if all probabilities are very low
        if np.max(probs) < safety_threshold:
            threshold = safety_threshold
        else:
            # 3. Apply threshold clamping (ensure it stays between 0.2 and 0.8)
            threshold = np.clip(threshold, 0.2, 0.8)

    # Apply threshold logic
    predictions = (probs > threshold).astype(int)
    num_anomalies = np.sum(predictions)
    
    # 7. Print threshold information
    print(f"Dynamic Threshold Used: {threshold:.4f}")
    print(f"Number of Anomalies Detected: {num_anomalies} (out of {data_size})")
    
    return float(threshold), predictions, probs

if __name__ == "__main__":
    # Example 1: Large batch (>50)
    large_probs = np.random.uniform(0, 0.3, 60)
    large_probs[10] = 0.9 # Add an anomaly
    print("--- Example 1: Large Batch (>50 samples) ---")
    thresh1, preds1, _ = compute_adaptive_threshold(large_probs)
    
    # Example 2: Medium batch (20-50)
    medium_probs = np.random.uniform(0, 0.3, 30)
    medium_probs[5] = 0.8
    print("\n--- Example 2: Medium Batch (30 samples) ---")
    thresh2, preds2, _ = compute_adaptive_threshold(medium_probs)
    
    # Example 3: Small batch (5-20)
    small_probs = np.array([0.1, 0.15, 0.2, 0.7, 0.12, 0.18])
    print("\n--- Example 3: Small Batch (6 samples) ---")
    thresh3, preds3, _ = compute_adaptive_threshold(small_probs)
    
    # Example 4: Very small batch (<5)
    tiny_probs = np.array([0.05, 0.9])
    print("\n--- Example 4: Very Small Batch (2 samples) ---")
    thresh4, preds4, _ = compute_adaptive_threshold(tiny_probs)
    
    # Example 5: Extremely clean data (safety logic)
    clean_probs = np.array([0.01, 0.02, 0.01, 0.03, 0.01, 0.02])
    print("\n--- Example 5: Extremely Clean Data (Safety logic) ---")
    thresh5, preds5, _ = compute_adaptive_threshold(clean_probs)
