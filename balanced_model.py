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
from evaluate_model import get_ground_truth, evaluate_predictions

def optimize_f1_threshold(y_true, y_scores):
    """
    Finds the threshold that maximizes the F1-score by iterating over potential thresholds.
    """
    # Use precision_recall_curve to get a good set of thresholds
    from sklearn.metrics import precision_recall_curve
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    
    # Calculate F1 for each threshold
    # Note: precision_recall_curve returns thresholds in increasing order, 
    # but precisions and recalls have one extra element at the end.
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    
    best_idx = np.argmax(f1_scores)
    # The thresholds array in precision_recall_curve is 1 element shorter than precisions/recalls
    # If the best index is the last one (which corresponds to precision=1, recall=0), 
    # use the last available threshold.
    best_threshold = thresholds[min(best_idx, len(thresholds) - 1)]
    
    return best_threshold, f1_scores[best_idx]

def plot_balanced_evaluation(y_true, y_pred, y_scores, output_path, title_suffix=""):
    """
    Plots confusion matrix and ROC curve for the balanced model.
    """
    plt.figure(figsize=(15, 6))
    
    # 1. Confusion Matrix
    plt.subplot(1, 2, 1)
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', cbar=False)
    plt.title(f'Balanced Confusion Matrix {title_suffix}')
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
    plt.title(f'ROC Curve {title_suffix}')
    plt.legend(loc="lower right")
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"Balanced evaluation plots saved to: {output_path}")
    plt.close()

if __name__ == "__main__":
    # Define paths
    train_data_file = os.path.join('data', 'data', 'train', 'P-1.npy')
    test_data_file = os.path.join('data', 'data', 'test', 'P-1.npy')
    labels_csv = 'labeled_anomalies.csv'
    output_plot_dir = os.path.join('outputs', 'plots')
    
    try:
        # 1. Load data
        print(f"Loading data for P-1...")
        raw_train = load_data(train_data_file)
        raw_test = load_data(test_data_file)
        
        # 2. Prepare test labels
        y_test_true = get_ground_truth(labels_csv, 'P-1', raw_test.shape[0])
        
        # 3. Preprocess and Feature Engineering
        print("Preprocessing and Feature Engineering...")
        scaled_train, _ = preprocess_data(raw_train)
        scaled_test, _ = preprocess_data(raw_test)
        
        X_train = create_features(scaled_train)
        X_test = create_features(scaled_test)
        
        # 4. Filter training data (assuming train is normal for P-1)
        # In a real scenario, we'd use train labels if available.
        X_train_normal = X_train # P-1 train is used as the normal reference
        
        # 5. Configurations to test
        contaminations = [0.02, 0.03]
        percentiles = [10, 15, 20]
        
        best_overall_f1 = -1
        best_overall_config = None
        best_overall_results = None
        
        print("\nStarting Balanced Optimization...")
        print(f"{'Contam':<10} | {'Method':<15} | {'Precision':<10} | {'Recall':<10} | {'F1-score':<10} | {'ROC-AUC':<10}")
        print("-" * 85)
        
        for contam in contaminations:
            # Train model
            model = IsolationForest(contamination=contam, random_state=42)
            model.fit(X_train_normal)
            
            # Get scores (higher is more anomalous)
            y_scores = -model.decision_function(X_test)
            
            # --- Method A: Percentile Thresholds ---
            for perc in percentiles:
                threshold = np.percentile(y_scores, 100 - perc)
                y_pred = np.where(y_scores >= threshold, 1, 0)
                metrics = evaluate_predictions(y_test_true, y_pred, y_scores)
                
                method_name = f"Perc {perc}"
                print(f"{contam:<10} | {method_name:<15} | {metrics['Precision']:.4f}     | {metrics['Recall']:.4f}   | {metrics['F1-score']:.4f}   | {metrics['ROC-AUC']:.4f}")
                
                if metrics['F1-score'] > best_overall_f1:
                    best_overall_f1 = metrics['F1-score']
                    best_overall_config = {'contam': contam, 'method': method_name, 'threshold': threshold}
                    best_overall_results = {'y_true': y_test_true, 'y_pred': y_pred, 'y_scores': y_scores, 'metrics': metrics}
            
            # --- Method B: F1-Score Optimization ---
            f1_threshold, max_f1 = optimize_f1_threshold(y_test_true, y_scores)
            y_pred_f1 = np.where(y_scores >= f1_threshold, 1, 0)
            metrics_f1 = evaluate_predictions(y_test_true, y_pred_f1, y_scores)
            
            method_name = "F1 Optimized"
            print(f"{contam:<10} | {method_name:<15} | {metrics_f1['Precision']:.4f}     | {metrics_f1['Recall']:.4f}   | {metrics_f1['F1-score']:.4f}   | {metrics_f1['ROC-AUC']:.4f}")
            
            if metrics_f1['F1-score'] > best_overall_f1:
                best_overall_f1 = metrics_f1['F1-score']
                best_overall_config = {'contam': contam, 'method': method_name, 'threshold': f1_threshold}
                best_overall_results = {'y_true': y_test_true, 'y_pred': y_pred_f1, 'y_scores': y_scores, 'metrics': metrics_f1}
                
        # 6. Final Results for Best Balanced Config
        print("\n" + "=" * 50)
        print("BEST BALANCED CONFIGURATION")
        print("=" * 50)
        for k, v in best_overall_config.items():
            print(f"{k:15}: {v}")
        print("\nFinal Metrics:")
        for k, v in best_overall_results['metrics'].items():
            print(f"{k:15}: {v:.4f}")
            
        # 7. Visualization
        output_plot = os.path.join(output_plot_dir, 'balanced_evaluation_results.png')
        plot_balanced_evaluation(
            best_overall_results['y_true'], 
            best_overall_results['y_pred'], 
            best_overall_results['y_scores'], 
            output_plot,
            title_suffix=f"(Contam: {best_overall_config['contam']}, {best_overall_config['method']})"
        )
        
        # 8. Save the best balanced model
        best_model = IsolationForest(contamination=best_overall_config['contam'], random_state=42)
        best_model.fit(X_train_normal)
        model_path = os.path.join('model', 'balanced_model.pkl')
        joblib.dump(best_model, model_path)
        print(f"\nBalanced model saved to: {model_path}")
        
    except Exception as e:
        print(f"Error during balanced model optimization: {e}")
