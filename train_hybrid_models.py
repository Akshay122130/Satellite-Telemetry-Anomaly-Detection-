import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from preprocess import load_data, preprocess_data
from features import create_features
from evaluate_model import get_ground_truth

def train_and_save_hybrid_components():
    # Define paths
    test_data_file = os.path.join('data', 'data', 'test', 'P-1.npy')
    labels_csv = 'labeled_anomalies.csv'
    model_dir = 'model'
    os.makedirs(model_dir, exist_ok=True)
    
    # 1. Load data for training
    # Using the P-1 data as per previous model comparison steps
    print("Loading data for P-1...")
    raw_data = load_data(test_data_file)
    y_full = get_ground_truth(labels_csv, 'P-1', raw_data.shape[0])
    
    # 2. Preprocess and Feature Engineering
    print("Preprocessing and Feature Engineering...")
    scaled_data, _ = preprocess_data(raw_data)
    X_full = create_features(scaled_data)
    
    # 3. Train models
    print("Training Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_full, y_full)
    
    print("Training XGBoost...")
    xgb_model = XGBClassifier(eval_metric='logloss', random_state=42)
    xgb_model.fit(X_full, y_full)
    
    # 4. Save models
    rf_path = os.path.join(model_dir, 'rf_model.pkl')
    xgb_path = os.path.join(model_dir, 'xgb_model.pkl')
    
    joblib.dump(rf_model, rf_path)
    joblib.dump(xgb_model, xgb_path)
    
    print(f"\nRandom Forest saved to: {rf_path}")
    print(f"XGBoost saved to:       {xgb_path}")

if __name__ == "__main__":
    train_and_save_hybrid_components()
