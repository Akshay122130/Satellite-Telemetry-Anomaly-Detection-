import numpy as np

class ContextProcessor:
    """
    Handles context-aware telemetry and probability processing to account for 
    environmental factors dynamically.
    
    Why Context is Important:
    1. Real-world sensors are affected by external environments (e.g., heat from sunlight, 
       cooling in shadows).
    2. These shifts can mimic anomalies (False Positives) or mask them (False Negatives).
    3. Context-aware processing improves accuracy by adjusting the baseline or the 
       confidence of the model based on known environmental states.
    """
    
    def __init__(self):
        # Base direction of impact for different contexts
        self.context_directions = {
            "sunlight": 1,   # Positive impact direction
            "shadow": -1,    # Negative impact direction
            "normal": 0      # No impact
        }

    def _compute_dynamic_weight(self, data):
        """
        Computes an impact weight based on data variability (Standard Deviation).
        Highly variable data gets a smaller relative context adjustment.
        """
        if isinstance(data, (int, float, np.number)):
            return 0.05 # Default 5% for single values
            
        std = np.std(data)
        # Normalize weight: more stable data (low std) allows for clearer context impact
        # We cap the weight to ensure it remains realistic (e.g., between 1% and 10%)
        base_weight = 0.05
        if std > 0:
            dynamic_factor = 1.0 / (1.0 + std)
            weight = np.clip(base_weight * dynamic_factor, 0.01, 0.10)
        else:
            weight = base_weight
            
        return weight

    def apply_context(self, data, context="normal", mode="data"):
        """
        Applies context-specific transformations to telemetry data or anomaly probabilities.
        
        Args:
            data (np.ndarray or float): The input data (raw values or probabilities).
            context (str): The environmental context ("sunlight", "shadow", "normal").
            mode (str): "data" to modify telemetry values, "probability" to modify anomaly scores.
            
        Returns:
            adjusted_output: The context-adjusted data or probability.
            applied_weight: The dynamic weight percentage used for the transformation.
        """
        is_single_value = isinstance(data, (int, float, np.number))
        data_array = np.array(data, dtype=float)
        
        # 1. Determine base impact direction and dynamic weight
        ctx = context.lower()
        direction = self.context_directions.get(ctx, self.context_directions["normal"])
        weight = self._compute_dynamic_weight(data)
        
        # 2. Perform adjustment based on mode
        if mode == "data":
            # Adjust only significant values (above the mean) to simulate realistic impact
            mean_val = np.mean(data_array)
            mask = data_array > mean_val
            
            # Apply adjustment: value * (1 + direction * weight)
            adjustment = 1.0 + (direction * weight)
            
            adjusted_output = data_array.copy()
            adjusted_output[mask] = data_array[mask] * adjustment
            
        elif mode == "probability":
            # In probability mode, sunlight increases risk, shadow decreases it
            # Shift probability slightly based on weight
            # Formula: prob + (direction * weight * (1 - prob if direction > 0 else prob))
            # This ensures probabilities stay within [0, 1]
            if direction > 0:
                # Sunlight: increase probability
                adjusted_output = data_array + (weight * (1.0 - data_array))
            elif direction < 0:
                # Shadow: decrease probability
                adjusted_output = data_array - (weight * data_array)
            else:
                adjusted_output = data_array
        else:
            raise ValueError(f"Invalid mode: {mode}. Use 'data' or 'probability'.")

        # 3. Finalize output
        if is_single_value:
            final_val = float(adjusted_output.item())
            return final_val, weight
        
        return adjusted_output, weight

if __name__ == "__main__":
    processor = ContextProcessor()
    
    # Sample batch data (e.g., temperatures)
    raw_telemetry = np.array([20.0, 22.0, 25.0, 19.0, 30.0])
    print(f"Original Telemetry: {raw_telemetry}")
    print(f"Mean Value:         {np.mean(raw_telemetry):.2f}")
    
    # 1. Data Adjustment Mode (Sunlight)
    adj_data, w1 = processor.apply_context(raw_telemetry, "sunlight", mode="data")
    print(f"\n--- Data Mode: Sunlight ---")
    print(f"Weight Applied: {w1:.4f}")
    print(f"Adjusted Data:  {adj_data}")
    print("(Only values above mean were increased)")
    
    # 2. Probability Adjustment Mode (Shadow)
    probs = np.array([0.1, 0.5, 0.8])
    adj_probs, w2 = processor.apply_context(probs, "shadow", mode="probability")
    print(f"\n--- Probability Mode: Shadow ---")
    print(f"Original Probs: {probs}")
    print(f"Weight Applied: {w2:.4f}")
    print(f"Adjusted Probs: {adj_probs}")
    print("(Probabilities were decreased slightly)")
    
    # 3. Single Value Probability Test
    single_p = 0.7
    adj_p, w3 = processor.apply_context(single_p, "sunlight", mode="probability")
    print(f"\n--- Single Prob: Sunlight ---")
    print(f"Original: {single_p} -> Adjusted: {adj_p:.4f} (Weight: {w3:.4f})")
