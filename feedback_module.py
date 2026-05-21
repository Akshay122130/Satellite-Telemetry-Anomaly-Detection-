import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ModelUpdater:
    """
    Handles model adaptation and self-learning by updating existing models 
    with new labeled telemetry data.
    
    Concept Drift Handling:
    - Concept drift occurs when the underlying data distribution changes over time.
    - By periodically updating the model with fresh labeled data, we ensure the 
      decision boundaries remain relevant to the current system state.
    - This module allows the model to "learn" from its mistakes (false positives/negatives) 
      once they are corrected by a human-in-the-loop or verified by ground truth.
    """

    def update_model(self, model, X_new, y_new):
        """
        Updates the existing model using new data and labels.
        
        Args:
            model: The currently trained model (RandomForestClassifier or XGBClassifier).
            X_new (np.ndarray): New feature data.
            y_new (np.ndarray): New labels for the feature data.
            
        Returns:
            updated_model: The model after adaptation.
        """
        model_type = type(model).__name__
        
        # Ensure data is in the correct format
        X_new = np.array(X_new)
        y_new = np.array(y_new)
        
        logging.info(f"Updating {model_type} with {len(y_new)} new samples...")

        if "RandomForestClassifier" in model_type:
            # For Random Forest, we use warm_start to add more trees trained on the new data
            # This allows incremental improvement without discarding old knowledge
            model.warm_start = True
            model.n_estimators += 10 # Add 10 new trees for the new batch
            model.fit(X_new, y_new)
            
        elif "XGBClassifier" in model_type:
            # For XGBoost, we can pass the existing model to the 'fit' method
            # to continue boosting from the current state
            model.fit(X_new, y_new, xgb_model=model.get_booster())
            
        else:
            # For other models, we might need a full retrain if they don't support incremental updates
            logging.warning(f"Incremental update not natively supported for {model_type}. Performing full retrain on new data.")
            model.fit(X_new, y_new)

        logging.info(f"Model update complete. New n_samples processed: {len(y_new)}")
        return model

if __name__ == "__main__":
    # 1. Simulate initial training
    X_init = np.random.rand(100, 5)
    y_init = np.random.randint(0, 2, 100)
    
    rf = RandomForestClassifier(n_estimators=10, random_state=42)
    rf.fit(X_init, y_init)
    
    xgb = XGBClassifier(n_estimators=10, random_state=42)
    xgb.fit(X_init, y_init)
    
    # 2. Simulate new data for self-learning
    X_new = np.random.rand(20, 5)
    y_new = np.random.randint(0, 2, 20)
    
    updater = ModelUpdater()
    
    # 3. Update Random Forest
    print("--- Testing RF Update ---")
    old_n_trees = rf.n_estimators
    updated_rf = updater.update_model(rf, X_new, y_new)
    print(f"RF trees: {old_n_trees} -> {updated_rf.n_estimators}")
    
    # 4. Update XGBoost
    print("\n--- Testing XGBoost Update ---")
    updated_xgb = updater.update_model(xgb, X_new, y_new)
    print("XGBoost update successful.")
