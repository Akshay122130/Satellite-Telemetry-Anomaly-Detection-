import numpy as np

def compute_health_score(probability):
    """
    Converts anomaly probability into a system health score.
    
    Interpretation:
    - 100: Perfect health (0% anomaly probability)
    - 0: Critical state (100% anomaly probability)
    - Higher score indicates better system stability and normal operation.
    - Lower score signals a potential anomaly or system degradation.
    
    Args:
        probability (np.ndarray or float): Final anomaly probability (0.0 to 1.0).
        
    Returns:
        np.ndarray or float: Health score (0 to 100).
    """
    # Ensure input is a numpy array for consistent batch processing
    is_single_value = isinstance(probability, (int, float, np.number))
    prob_array = np.array(probability)
    
    # Logic: health_score = (1 - probability) * 100
    # We clip to ensure the result is strictly between 0 and 100
    health_score = np.clip((1.0 - prob_array) * 100.0, 0.0, 100.0)
    
    if is_single_value:
        return float(health_score.item())
    
    return health_score

if __name__ == "__main__":
    # Example 1: Single probability (Normal)
    p1 = 0.05
    score1 = compute_health_score(p1)
    print(f"--- Single Value: Normal ---")
    print(f"Probability: {p1:.4f} -> Health Score: {score1:.2f}/100")
    
    # Example 2: Single probability (Anomaly)
    p2 = 0.85
    score2 = compute_health_score(p2)
    print(f"\n--- Single Value: Anomaly ---")
    print(f"Probability: {p2:.4f} -> Health Score: {score2:.2f}/100")
    
    # Example 3: Batch of probabilities
    batch_probs = np.array([0.01, 0.12, 0.50, 0.95])
    batch_scores = compute_health_score(batch_probs)
    print(f"\n--- Batch Input ---")
    print(f"Probabilities: {batch_probs}")
    print(f"Health Scores: {batch_scores}")
    
    # Example 4: Edge case (Outside [0, 1])
    edge_p = np.array([-0.1, 1.2])
    edge_scores = compute_health_score(edge_p)
    print(f"\n--- Edge Case (Clamping) ---")
    print(f"Input:  {edge_p}")
    print(f"Output: {edge_scores}")
