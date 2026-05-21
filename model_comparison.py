import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, roc_curve, auc)
import joblib

from preprocess import load_data, preprocess_data
from features import create_features
from evaluate_model import get_ground_truth

def evaluate_model(model, X_test, y_test, is_unsupervised=False):
    """
    Evaluates a model and returns metrics and ROC curve data.
    """
    if is_unsupervised:
        # For Isolation Forest
        # Predict: -1 for anomaly, 1 for normal
        preds_raw = model.predict(X_test)
        y_pred = np.where(preds_raw == -1, 1, 0)
        # Scores: decision_function (lower is more anomalous)
        # We want higher for anomaly for ROC-AUC
        y_scores = -model.decision_function(X_test)
    else:
        # For supervised models
        y_pred = model.predict(X_test)
        y_scores = model.predict_proba(X_test)[:, 1]
        
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred, zero_division=0),
        'F1-score': f1_score(y_test, y_pred, zero_division=0),
        'ROC-AUC': roc_auc_score(y_test, y_scores)
    }
    
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    
    return metrics, fpr, tpr

def run_model_comparison():
    # Define paths
    test_data_file = os.path.join('data', 'data', 'test', 'P-1.npy')
    labels_csv = 'labeled_anomalies.csv'
    output_dir = os.path.join('outputs', 'model_comparison')
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load data and labels for P-1
    print("Loading data and labels for P-1...")
    raw_data = load_data(test_data_file)
    y_full = get_ground_truth(labels_csv, 'P-1', raw_data.shape[0])
    
    # 2. Preprocess and Feature Engineering
    print("Preprocessing and Feature Engineering...")
    scaled_data, _ = preprocess_data(raw_data)
    X_full = create_features(scaled_data)
    
    # 3. Split data (80/20)
    print(f"Splitting data into 80% Train and 20% Test...")
    X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.2, random_state=42, stratify=y_full)
    
    # 4. Models to train
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        'Isolation Forest': IsolationForest(contamination=0.05, random_state=42)
    }
    
    results_list = []
    roc_curves = {}
    
    print("\nTraining and Evaluating Models...")
    for name, model in models.items():
        print(f"Running {name}...")
        
        is_unsupervised = (name == 'Isolation Forest')
        
        if is_unsupervised:
            # Train only on training set for comparison, but it's unsupervised
            model.fit(X_train)
        else:
            # Supervised training
            model.fit(X_train, y_train)
            
        metrics, fpr, tpr = evaluate_model(model, X_test, y_test, is_unsupervised=is_unsupervised)
        
        metrics['Model'] = name
        results_list.append(metrics)
        roc_curves[name] = (fpr, tpr)
        
    # 5. Store results in a table
    results_df = pd.DataFrame(results_list)
    results_df = results_df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-score', 'ROC-AUC']]
    
    print("\n--- Model Comparison Table ---")
    print(results_df.to_string(index=False))
    
    # Save table
    results_df.to_csv(os.path.join(output_dir, 'comparison_metrics.csv'), index=False)
    
    # 6. Visualizations
    # Bar Chart
    plt.figure(figsize=(12, 6))
    melted_df = results_df.melt(id_vars='Model', var_name='Metric', value_name='Value')
    sns.barplot(data=melted_df, x='Metric', y='Value', hue='Model')
    plt.title('Model Performance Comparison')
    plt.ylim(0, 1.1)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, 'performance_bar_chart.png'))
    plt.close()
    
    # ROC Curves
    plt.figure(figsize=(10, 8))
    for name, (fpr, tpr) in roc_curves.items():
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.4f})')
        
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(output_dir, 'roc_curves_comparison.png'))
    plt.close()
    
    # 7. Best Model based on F1-score
    best_model_row = results_df.loc[results_df['F1-score'].idxmax()]
    print("\n" + "="*40)
    print(f"BEST MODEL: {best_model_row['Model']}")
    print(f"F1-score:   {best_model_row['F1-score']:.4f}")
    print("="*40)
    
    print(f"\nAll results and visualizations saved to: {output_dir}")

if __name__ == "__main__":
    try:
        run_model_comparison()
    except Exception as e:
        print(f"Error during model comparison: {e}")
