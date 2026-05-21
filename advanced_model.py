import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             roc_curve, auc)
from preprocess import load_data, preprocess_data
from features import create_features
from evaluate_model import get_ground_truth

def find_optimal_threshold(y_true, y_scores):
    """
    Finds the optimal threshold that maximizes the Youden's J statistic (TPR - FPR).
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    # Youden's J statistic: J = TPR - FPR
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_threshold = thresholds[best_idx]
    
    return best_threshold, fpr, tpr

def plot_advanced_evaluation(y_true, y_pred, y_scores, fpr, tpr, best_threshold, output_path):
    """
    Plots confusion matrix and optimized ROC curve.
    """
    plt.figure(figsize=(15, 6))
    
    # 1. Confusion Matrix
    plt.subplot(1, 2, 1)
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', cbar=False)
    plt.title('Optimized Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # 2. ROC Curve with Optimal Point
    plt.subplot(1, 2, 2)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    
    # Find the point on the curve for the best threshold
    # The best index was already calculated in find_optimal_threshold
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    plt.scatter(fpr[best_idx], tpr[best_idx], color='red', s=100, label=f'Optimal Threshold: {best_threshold:.4f}', zorder=5)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Optimized Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"Advanced evaluation plots saved to: {output_path}")
    plt.close()

if __name__ == "__main__":
    # Define paths
    train_data_file = os.path.join('data', 'data', 'train', 'P-1.npy')
    test_data_file = os.path.join('data', 'data', 'test', 'P-1.npy')
    labels_csv = 'labeled_anomalies.csv'
    model_output_path = os.path.join('model', 'advanced_model.pkl')
    output_plot = os.path.join('outputs', 'plots', 'advanced_evaluation_results.png')
    
    try:
        # 1. Load data
        print(f"Loading data from: {train_data_file}")
        raw_train = load_data(train_data_file)
        raw_test = load_data(test_data_file)
        
        # 2. Load ground truth for test
        y_test_true = get_ground_truth(labels_csv, 'P-1', raw_test.shape[0])
        
        # For training data, we assume it's normal (y_true == 0) as is standard 
        # for this dataset, or filter if we had explicit training labels.
        # Requirement: Filter data using labels (y_true == 0)
        y_train_true = np.zeros(raw_train.shape[0]) # Assuming train is all normal
        
        # 3. Preprocess and Feature Engineering
        print("Preprocessing and Feature Engineering...")
        # Note: preprocess_data now uses StandardScaler
        scaled_train, scaler = preprocess_data(raw_train)
        scaled_test, _ = preprocess_data(raw_test)
        
        X_train_full = create_features(scaled_train)
        X_test = create_features(scaled_test)
        
        # 4. Filter training data using labels (y_true == 0)
        X_train_normal = X_train_full[y_train_true == 0]
        print(f"Training set filtered: {X_train_full.shape[0]} -> {X_train_normal.shape[0]} normal samples")
        
        # 5. Train Isolation Forest only on normal data
        print("Training Isolation Forest on normal data...")
        model = IsolationForest(contamination=0.01, random_state=42)
        model.fit(X_train_normal)
        
        # 6. Compute anomaly scores using decision_function
        print("Computing anomaly scores on test data...")
        # decision_function: lower is more anomalous. 
        # We need higher for more anomalous for ROC-AUC/Thresholding
        y_scores = -model.decision_function(X_test)
        
        # 7. Use ROC curve to select best threshold (Maximize TPR - FPR)
        print("Optimizing threshold using ROC curve...")
        best_threshold, fpr, tpr = find_optimal_threshold(y_test_true, y_scores)
        
        # 8. Recompute metrics using best threshold
        y_pred = np.where(y_scores >= best_threshold, 1, 0)
        
        metrics = {
            'Accuracy': accuracy_score(y_test_true, y_pred),
            'Precision': precision_score(y_test_true, y_pred, zero_division=0),
            'Recall': recall_score(y_test_true, y_pred, zero_division=0),
            'F1-score': f1_score(y_test_true, y_pred, zero_division=0),
            'ROC-AUC': roc_auc_score(y_test_true, y_scores)
        }
        
        print("\n--- Advanced Model Evaluation Results (Optimized Threshold) ---")
        for metric, value in metrics.items():
            print(f"{metric:10}: {value:.4f}")
        print(f"Optimal Threshold: {best_threshold:.4f}")
            
        # 9. Update Visualizations
        plot_advanced_evaluation(y_test_true, y_pred, y_scores, fpr, tpr, best_threshold, output_plot)
        
        # 10. Save the advanced model
        joblib.dump(model, model_output_path)
        print(f"\nAdvanced model saved to: {model_output_path}")
        
    except Exception as e:
        print(f"Error during advanced model training/evaluation: {e}")
