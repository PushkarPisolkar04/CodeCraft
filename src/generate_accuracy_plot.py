import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
import joblib
import os

def load_test_data():
    """Load preprocessed test data."""
    test_data = pd.read_csv('data/test.csv')
    
    # The features are already preprocessed
    feature_cols = ['Average_Cost', 'Minimum_Order', 'Rating', 'Votes', 'Reviews',
                   'City', 'Cuisine_Count', 'Has_North_Indian', 'Has_South_Indian',
                   'Has_Chinese', 'Has_Fast_Food', 'Price_Category']
    
    X_test = test_data[feature_cols]
    
    # Convert delivery categories to numeric classes
    y_test = test_data['Delivery_Category'].map({
        'Fast (≤30 mins)': 0,
        'Medium (31-45 mins)': 1,
        'Slow (>45 mins)': 2
    })
    
    return X_test, y_test

def evaluate_models():
    """Evaluate all models and return their accuracies."""
    # Load models
    models = {
        'Random Forest': joblib.load('models/random_forest_model.joblib'),
        'XGBoost': joblib.load('models/xgboost_model.joblib'),
        'LightGBM': joblib.load('models/lightgbm_model.joblib'),
        'Logistic Regression': joblib.load('models/logistic_regression_model.joblib'),
        'SVC': joblib.load('models/svc_model.joblib'),
        'KNN': joblib.load('models/knn_model.joblib'),
        'Decision Tree': joblib.load('models/decision_tree_model.joblib')
    }
    
    # Load test data
    X_test, y_test = load_test_data()
    
    # Calculate accuracies
    accuracies = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred) * 100
        accuracies[name] = acc
    
    return accuracies

def create_accuracy_plot():
    """Create and save the model accuracy comparison plot."""
    # Get accuracies
    accuracies = evaluate_models()
    
    # Sort accuracies
    accuracies = dict(sorted(accuracies.items(), key=lambda x: x[1], reverse=True))
    
    # Set style
    sns.set_style("whitegrid")
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Create bar plot with custom colors
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f1c40f', '#1abc9c', '#e67e22']
    bars = plt.bar(accuracies.keys(), accuracies.values(), color=colors)
    
    # Customize plot
    plt.title('Model Accuracy Comparison', fontsize=14, pad=20)
    plt.xlabel('Models', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    
    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%',
                ha='center', va='bottom',
                fontsize=10,
                fontweight='bold')
    
    # Add grid for better readability
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Set y-axis limits
    plt.ylim(0, max(accuracies.values()) * 1.1)
    
    # Adjust layout
    plt.tight_layout()
    
    # Create static directory if it doesn't exist
    os.makedirs('static', exist_ok=True)
    
    # Save plot with white background
    plt.savefig('static/model_accuracy.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

if __name__ == "__main__":
    create_accuracy_plot() 