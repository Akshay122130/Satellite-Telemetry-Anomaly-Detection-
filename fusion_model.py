import os
import joblib
import numpy as np

class FusionModel:
    """
    Hybrid Anomaly Detection Model using Weighted Fusion of 
    Random Forest and XGBoost.
    """
    def __init__(self, rf_path='model/rf_model.pkl', xgb_path='model/xgb_model.pkl'):
        # 1. Load trained models
        if not os.path.exists(rf_path):
            raise FileNotFoundError(f"Random Forest model not found: {rf_path}")
        if not os.path.exists(xgb_path):
            raise FileNotFoundError(f"XGBoost model not found: {xgb_path}")
            
        self.rf_model = joblib.load(rf_path)
        self.xgb_model = joblib.load(xgb_path)
        print("Models loaded successfully.")

    def predict(self, features):
        """
        Generate weighted fusion predictions.
        
        Args:
            features (np.ndarray): Input feature array (n_samples, n_features)
            
        Returns:
            final_prob (np.ndarray): Weighted probabilities
            final_pred (list): List of predictions ("Anomaly" or "Normal")
            individual_probs (dict): Dictionary of individual model probabilities
        """
        # Ensure features is a numpy array
        features = np.array(features)
        
        # Handle single input vs batch input
        # If features is 1D, reshape it to (1, n_features)
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
            
        # 3. Generate probabilities (predict_proba)
        # RF and XGB both return probabilities for classes [0, 1]
        # We take the probability of class 1 (Anomaly)
        rf_prob = self.rf_model.predict_proba(features)[:, 1]
        xgb_prob = self.xgb_model.predict_proba(features)[:, 1]
        
        # 4. Implement weighted fusion: 
        # final_prob = 0.7 * rf_prob + 0.3 * xgb_prob
        # Logic: Random Forest is given higher weight (0.7) because it showed superior 
        # F1-score and lower false positive rates in individual model evaluations.
        final_prob = 0.7 * rf_prob + 0.3 * xgb_prob
        
        # 5. Decision rule: 
        # Threshold is reduced to 0.4 to improve detection sensitivity (Recall).
        # We also implement a safety rule: if RF confidence > 0.6, it's an anomaly 
        # to ensure critical events are not missed due to ensemble averaging.
        final_pred = []
        for i in range(len(final_prob)):
            if rf_prob[i] > 0.6:
                final_pred.append("Anomaly") # Safety rule override
            elif final_prob[i] > 0.4:
                final_pred.append("Anomaly") # Lowered threshold decision
            else:
                final_pred.append("Normal")
        
        # Store individual model probabilities for optional analysis
        individual_probs = {
            'rf_prob': rf_prob,
            'xgb_prob': xgb_prob
        }
        
        return final_prob, final_pred, individual_probs

if __name__ == "__main__":
    from preprocess import load_data, preprocess_data
    from features import create_features
    from evaluate_model import get_ground_truth
    
    # Example usage with P-1 test data
    test_data_file = os.path.join('data', 'data', 'test', 'P-1.npy')
    labels_csv = 'labeled_anomalies.csv'
    
    try:
        # Load and prepare sample data
        raw_data = load_data(test_data_file)
        y_true = get_ground_truth(labels_csv, 'P-1', raw_data.shape[0])
        
        scaled_data, _ = preprocess_data(raw_data)
        X_full = create_features(scaled_data)
        
        # 2. Initialize Fusion Model
        fusion = FusionModel()
        
        # Test with a single sample (the first anomaly, if it exists)
        # P-1 has an anomaly starting at 2149
        sample_index = 2150 
        sample_features = X_full[sample_index]
        
        print(f"\n--- Single Sample Prediction (Index {sample_index}) ---")
        prob, pred, details = fusion.predict(sample_features)
        print(f"True Label:     {'Anomaly' if y_true[sample_index] == 1 else 'Normal'}")
        print(f"RF Prob:        {details['rf_prob'][0]:.4f}")
        print(f"XGB Prob:       {details['xgb_prob'][0]:.4f}")
        print(f"Final Prob:     {prob[0]:.4f}")
        print(f"Final Prediction: {pred[0]}")
        
        # Test with a batch of samples
        print(f"\n--- Batch Prediction (First 5 samples) ---")
        probs, preds, details = fusion.predict(X_full[:5])
        for i, (p, pr) in enumerate(zip(probs, preds)):
            print(f"Sample {i}: RF={details['rf_prob'][i]:.4f}, XGB={details['xgb_prob'][i]:.4f}, Final={p:.4f}, Pred={pr}")
            
    except Exception as e:
        print(f"Error during fusion model testing: {e}")
