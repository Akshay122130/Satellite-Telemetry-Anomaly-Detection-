import numpy as np
import os
import joblib

# Import custom modules
from preprocess import preprocess_data
from features import create_features
from fusion_model import FusionModel
from dynamic_threshold import compute_adaptive_threshold
from context_module import ContextProcessor
from health_score import compute_health_score

class AnomalyDetectionPipeline:
    """
    Integrated Anomaly Detection Pipeline for Structured Telemetry.
    
    Flow:
    1. Input: temperature, voltage, current, battery, context
    2. Preprocessing & Feature Engineering
    3. Model Inference (RF & XGBoost)
    4. Fusion (Weighted Ensemble)
    5. Context-Aware Probability Adjustment
    6. Dynamic Thresholding
    7. Health Scoring
    """
    
    def __init__(self, rf_path='model/rf_model.pkl', xgb_path='model/xgb_model.pkl'):
        self.fusion_model = FusionModel(rf_path=rf_path, xgb_path=xgb_path)
        self.context_processor = ContextProcessor()
        print("Pipeline initialized with structured support.")

    def run(self, temperature, voltage, current, battery, context="normal"):
        """
        Runs the complete pipeline for a single set of structured telemetry inputs.
        """
        # 2. Convert input into feature array
        # Shape: (1, 4)
        raw_features = np.array([[temperature, voltage, current, battery]])
        
        # 3. Apply Preprocessing and Feature Engineering
        # Preprocessing (StandardScaler)
        scaled_data, _ = preprocess_data(raw_features)
        
        # Feature Engineering (Expanding to 20 features: original + rolling/lag/std/diff)
        # Note: For single input, rolling/diff will use bfill/fillna(0) logic in features.py
        feature_matrix = create_features(scaled_data)
        
        # 4 & 5. Generate predictions and apply fusion model
        # Returns fused probability (0.7 RF + 0.3 XGB)
        final_prob, _, _ = self.fusion_model.predict(feature_matrix)
        
        # 5. Apply Context Module (Probability Mode)
        # Adjusts the anomaly probability based on environmental context (sunlight/shadow)
        adjusted_prob, ctx_weight = self.context_processor.apply_context(
            final_prob, context=context, mode="probability"
        )
        
        # 5. Apply Dynamic Threshold
        # Determines if the adjusted probability exceeds the adaptive boundary
        threshold_val, binary_predictions, _ = compute_adaptive_threshold(adjusted_prob)
        
        # 5. Apply Health Score Module
        # Converts the final adjusted probability into a 0-100 health index
        health_score = compute_health_score(adjusted_prob)
        
        # Map binary prediction to label
        prediction = "Anomaly" if binary_predictions[0] == 1 else "Normal"
        
        # 6. Return final output
        return {
            "prediction": prediction,
            "probability": float(adjusted_prob[0]),
            "threshold": float(threshold_val),
            "health_score": float(health_score[0] if isinstance(health_score, np.ndarray) else health_score)
        }

if __name__ == "__main__":
    # Example integration test with single structured input
    pipeline = AnomalyDetectionPipeline()
    
    # Simulate a single set of telemetry readings
    test_input = {
        "temperature": 25.5,
        "voltage": 28.2,
        "current": 1.5,
        "battery": 85.0,
        "context": "sunlight"
    }
    
    print("\n--- Testing Structured Pipeline (Single Input) ---")
    results = pipeline.run(**test_input)
    
    print(f"Input Context: {test_input['context']}")
    print(f"Prediction:    {results['prediction']}")
    print(f"Probability:   {results['probability']:.4f}")
    print(f"Threshold:     {results['threshold']:.4f}")
    print(f"Health Score:  {results['health_score']:.2f}/100")
