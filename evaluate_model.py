import os
import numpy as np
import pandas as pd
import joblib
import ast
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             roc_curve, auc)
from preprocess import load_data, preprocess_data
from features import create_features

def get_ground_truth(csv_path, chan_id, num_samples):
    """
    Extracts ground truth labels for a specific channel from the CSV file.
    """
    df_labels = pd.read_csv(csv_path)
    row = df_labels[df_labels['chan_id'] == chan_id]
    
    if row.empty:
        raise ValueError(f"Channel {chan_id} not found in {csv_path}")
    
    # Parse anomaly sequences
    anomaly_sequences = ast.literal_eval(row['anomaly_sequences'].values[0])
    
    # Create binary labels (0: normal, 1: anomaly)
    labels = np.zeros(num_samples, dtype=int)
    for start, end in anomaly_sequences:
        labels[start:end] = 1
        
    return labels

def evaluate_predictions(y_true, y_pred, y_scores):
    """
    Computes classification metrics.
    """
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1-score': f1_score(y_true, y_pred, zero_division=0),
        'ROC-AUC': roc_auc_score(y_true, y_scores)
    }
    return metrics

def plot_evaluation(y_true, y_pred, y_scores, output_path):
    """
    Plots confusion matrix and ROC curve.
    """
    plt.figure(figsize=(15, 6))
    
    # 1. Confusion Matrix
    plt.subplot(1, 2, 1)
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # 2. ROC Curve
    plt.subplot(1, 2, 2)
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"Evaluation plots saved to: {output_path}")
    plt.close()

if __name__ == "__main__":
    # Define paths
    test_data_file = os.path.join('data', 'data', 'test', 'P-1.npy')
    model_path = os.path.join('model', 'model.pkl')
    labels_csv = 'labeled_anomalies.csv'
    output_plot = os.path.join('outputs', 'plots', 'evaluation_results.png')
    
    try:
        # 1. Load trained model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}. Run train_model.py first.")
        model = joblib.load(model_path)
        
        # 2. Load test data
        raw_test_data = load_data(test_data_file)
        num_samples = raw_test_data.shape[0]
        
        # 3. Load ground truth labels
        y_true = get_ground_truth(labels_csv, 'P-1', num_samples)
        
        # 4. Preprocess and feature engineer test data
        scaled_test_data, _ = preprocess_data(raw_test_data)
        X_test = create_features(scaled_test_data)
        
        # 5. Predict using Isolation Forest
        # Isolation Forest: -1 for anomaly, 1 for normal
        preds_raw = model.predict(X_test)
        y_pred = np.where(preds_raw == -1, 1, 0)
        
        # 6. Compute anomaly scores (for ROC-AUC)
        # decision_function: lower is more anomalous. 
        # We need higher for more anomalous for ROC-AUC
        y_scores = -model.decision_function(X_test)
        
        # 7. Compute metrics
        metrics = evaluate_predictions(y_true, y_pred, y_scores)
        
        print("\n--- Model Evaluation Results (P-1 Test Data) ---")
        for metric, value in metrics.items():
            print(f"{metric:10}: {value:.4f}")
            
        # 8. Visualizations
        plot_evaluation(y_true, y_pred, y_scores, output_plot)
        
    except Exception as e:
        print(f"Error during model evaluation: {e}")
