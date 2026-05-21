import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score)

# Import custom modules
from data_preparation import prepare_multi_sensor_data
from preprocess import preprocess_data
from features import create_features
from evaluate_model import get_ground_truth

def evaluate_and_print(name, y_test, y_pred, y_prob):
    """
    Evaluates model performance and prints metrics.
    """
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred, zero_division=0),
        'F1-score': f1_score(y_test, y_pred, zero_division=0),
        'ROC-AUC': roc_auc_score(y_test, y_prob)
    }
    
    print(f"\n--- {name} Results ---")
    for k, v in metrics.items():
        print(f"{k:10}: {v:.4f}")
    return metrics

def train_and_evaluate():
    # Define paths
    # Using test data because it matches the labels in labeled_anomalies.csv (8505 samples)
    data_file = os.path.join('data', 'data', 'test', 'P-1.npy')
    labels_csv = 'labeled_anomalies.csv'
    model_dir = 'model'
    os.makedirs(model_dir, exist_ok=True)
    
    try:
        # 1 & 2. Load dataset using data_preparation (Structured 4-feature matrix)
        print(f"Loading and preparing structured data from: {data_file}")
        X_raw = prepare_multi_sensor_data(data_file)
        
        # Apply preprocessing and feature engineering
        print("Applying preprocessing and feature engineering...")
        X_scaled, _ = preprocess_data(X_raw)
        X = create_features(X_scaled)
        
        # 3. Load labels
        print("Loading anomaly labels...")
        y = get_ground_truth(labels_csv, 'P-1', X.shape[0])
        
        # 4. Split data (80% Train, 20% Test)
        print(f"Splitting data (80/20)...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 5. Train models
        print(f"\nTraining models on engineered features (shape: {X.shape})...")
        
        # Random Forest
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        
        # XGBoost
        xgb_model = XGBClassifier(eval_metric='logloss', random_state=42)
        xgb_model.fit(X_train, y_train)
        
        # 6. Evaluate
        # RF Evaluation
        rf_pred = rf_model.predict(X_test)
        rf_prob = rf_model.predict_proba(X_test)[:, 1]
        evaluate_and_print("Random Forest", y_test, rf_pred, rf_prob)
        
        # XGBoost Evaluation
        xgb_pred = xgb_model.predict(X_test)
        xgb_prob = xgb_model.predict_proba(X_test)[:, 1]
        evaluate_and_print("XGBoost", y_test, xgb_pred, xgb_prob)
        
        # 8. Save models
        rf_path = os.path.join(model_dir, 'rf_model.pkl')
        xgb_path = os.path.join(model_dir, 'xgb_model.pkl')
        
        joblib.dump(rf_model, rf_path)
        joblib.dump(xgb_model, xgb_path)
        
        print(f"\nModels successfully saved to {model_dir}/")
        print(f"Feature matrix shape: {X.shape}")
        print("Training and evaluation complete.")
        
    except Exception as e:
        print(f"Error during training: {e}")

if __name__ == "__main__":
    train_and_evaluate()
