import os
import numpy as np
import matplotlib.pyplot as plt
import joblib
from preprocess import load_data, preprocess_data
from features import create_features

def compute_anomaly_scores(model, feature_matrix):
    """
    Computes anomaly scores from the Isolation Forest model.
    Using the negative of the decision function so that higher scores are more anomalous.
    """
    # Isolation Forest decision_function: lower is more anomalous
    # We take the negative so that higher is more anomalous
    scores = -model.decision_function(feature_matrix)
    return scores

def visualize_model_scores(scores, threshold, output_path):
    """
    Plots anomaly scores, the threshold line, and highlights anomalies.
    """
    plt.figure(figsize=(15, 8))
    
    # 1. Plot the anomaly scores
    plt.plot(scores, label='Anomaly Score', color='teal', alpha=0.7, linewidth=1)
    
    # 2. Draw the threshold line
    plt.axhline(y=threshold, color='red', linestyle='--', label=f'Threshold (Mean + 2*Std = {threshold:.4f})')
    
    # 3. Highlight anomalies (scores > threshold)
    anomaly_indices = np.where(scores > threshold)[0]
    anomaly_values = scores[anomaly_indices]
    
    if len(anomaly_indices) > 0:
        plt.scatter(anomaly_indices, anomaly_values, color='crimson', label='Anomalies', s=15, zorder=5)
    
    plt.title('Isolation Forest Anomaly Scores and Threshold')
    plt.xlabel('Time Step')
    plt.ylabel('Anomaly Score (Higher is more anomalous)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plt.savefig(output_path)
    print(f"Model visualization plot saved to: {output_path}")
    print(f"Total anomalies above threshold: {len(anomaly_indices)}")
    plt.close()

if __name__ == "__main__":
    # Define paths
    model_path = os.path.join('model', 'model.pkl')
    data_file = os.path.join('data', 'data', 'train', 'P-1.npy')
    output_plot = os.path.join('outputs', 'plots', 'model_scores.png')
    
    try:
        # 1. Load trained model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}. Please run train_model.py first.")
            
        print(f"Loading model from: {model_path}")
        model = joblib.load(model_path)
        
        # 2. Load and process data
        print(f"Processing data from: {data_file}")
        raw_data = load_data(data_file)
        scaled_data, _ = preprocess_data(raw_data)
        feature_matrix = create_features(scaled_data)
        
        # 3. Compute anomaly scores
        print("Computing anomaly scores...")
        scores = compute_anomaly_scores(model, feature_matrix)
        
        # 4. Compute threshold: mean + 2 * standard deviation
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        threshold = mean_score + 2 * std_score
        
        print(f"Mean Score: {mean_score:.4f}, Std Score: {std_score:.4f}")
        print(f"Calculated Threshold: {threshold:.4f}")
        
        # 5. Visualize and save plot
        visualize_model_scores(scores, threshold, output_plot)
        
    except Exception as e:
        print(f"Error during model visualization: {e}")
