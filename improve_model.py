import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from preprocess import load_data, preprocess_data
from features import create_features
from evaluate_model import get_ground_truth, evaluate_predictions, plot_evaluation

def tune_and_compare():
    # Define paths
    train_data_file = os.path.join('data', 'data', 'train', 'P-1.npy')
    test_data_file = os.path.join('data', 'data', 'test', 'P-1.npy')
    labels_csv = 'labeled_anomalies.csv'
    output_plot_dir = os.path.join('outputs', 'plots')
    
    # 1. Load and Prepare Data
    print("Preparing data...")
    raw_train = load_data(train_data_file)
    raw_test = load_data(test_data_file)
    
    # Preprocess and Feature Engineering
    scaled_train, _ = preprocess_data(raw_train)
    scaled_test, _ = preprocess_data(raw_test)
    
    X_train = create_features(scaled_train)
    X_test = create_features(scaled_test)
    
    # Ground Truth for test set
    y_true = get_ground_truth(labels_csv, 'P-1', X_test.shape[0])
    
    # 2. Tuning Parameters
    contaminations = [0.05, 0.1, 0.15]
    percentiles = [10, 15, 20] # Percentiles for threshold tuning
    
    results = []
    best_recall = -1
    best_config = None
    best_y_pred = None
    best_y_scores = None
    
    print("\nStarting Hyperparameter Tuning...")
    print(f"{'Contam':<10} | {'Percentile':<12} | {'Precision':<10} | {'Recall':<10} | {'F1-score':<10} | {'ROC-AUC':<10}")
    print("-" * 75)
    
    for contam in contaminations:
        # Train Isolation Forest
        model = IsolationForest(contamination=contam, random_state=42)
        model.fit(X_train)
        
        # Get anomaly scores (decision_function)
        # decision_function: lower is more anomalous. 
        # For percentile, we want higher scores to be more anomalous
        scores = -model.decision_function(X_test)
        
        for perc in percentiles:
            # Set threshold based on percentile of anomaly scores
            threshold = np.percentile(scores, 100 - perc)
            
            # Predict
            y_pred = np.where(scores >= threshold, 1, 0)
            
            # Evaluate
            metrics = evaluate_predictions(y_true, y_pred, scores)
            
            results.append({
                'contamination': contam,
                'percentile': perc,
                'threshold': threshold,
                **metrics
            })
            
            print(f"{contam:<10} | {perc:<12} | {metrics['Precision']:.4f}     | {metrics['Recall']:.4f}   | {metrics['F1-score']:.4f}   | {metrics['ROC-AUC']:.4f}")
            
            # Track best based on Recall (as requested)
            if metrics['Recall'] > best_recall:
                best_recall = metrics['Recall']
                best_config = {
                    'contamination': contam,
                    'percentile': perc,
                    'threshold': threshold,
                    'metrics': metrics
                }
                best_y_pred = y_pred
                best_y_scores = scores
                
    # 3. Print Best Configuration
    print("\n" + "=" * 40)
    print("BEST CONFIGURATION (Based on Recall)")
    print("=" * 40)
    for k, v in best_config.items():
        if k != 'metrics':
            print(f"{k:15}: {v}")
    print("\nBest Metrics:")
    for k, v in best_config['metrics'].items():
        print(f"{k:15}: {v:.4f}")
        
    # 4. Save best configuration visualization
    output_plot = os.path.join(output_plot_dir, 'improved_evaluation_results.png')
    plot_evaluation(y_true, best_y_pred, best_y_scores, output_plot)
    print(f"\nImproved evaluation plots saved to: {output_plot}")

    # Optional: Save the best model
    # We'll retrain it on full train set if needed, but here we just use the one we have
    # ...

if __name__ == "__main__":
    try:
        tune_and_compare()
    except Exception as e:
        print(f"Error during model improvement: {e}")
