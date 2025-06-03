import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def load_test_data():
    """Load preprocessed test data"""
    try:
        # Load test data
        test_data = pd.read_csv('data/test.csv')
        
        # Split features and target
        X_test = test_data.drop('Delivery_Category', axis=1)
        
        # Convert delivery category to numeric
        def category_to_class(category):
            if category == 'Fast (≤30 mins)':
                return 0
            elif category == 'Medium (31-45 mins)':
                return 1
            else:
                return 2
                
        y_test = test_data['Delivery_Category'].apply(category_to_class)
        
        return X_test, y_test
        
    except Exception as e:
        print(f"Error loading test data: {str(e)}")
        raise

def evaluate_models():
    """Evaluate all trained models and print their accuracy scores"""
    try:
        # Load models
        models = {
            'Logistic Regression': joblib.load('models/logistic_regression_model.joblib'),
            'Random Forest': joblib.load('models/random_forest_model.joblib'),
            'XGBoost': joblib.load('models/xgboost_model.joblib'),
            'LightGBM': joblib.load('models/lightgbm_model.joblib'),
            'SVC': joblib.load('models/svc_model.joblib'),
            'KNN': joblib.load('models/knn_model.joblib'),
            'Decision Tree': joblib.load('models/decision_tree_model.joblib')
        }
        
        # Load test data
        X_test, y_test = load_test_data()
        
        # Evaluate each model
        results = {}
        all_predictions = {}
        
        print("\nModel Accuracy Scores:")
        print("-" * 50)
        
        for name, model in models.items():
            try:
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Calculate accuracy
                accuracy = accuracy_score(y_test, y_pred)
                results[name] = accuracy
                all_predictions[name] = y_pred
                
                # Print accuracy
                print(f"{name:20} : {accuracy:.4f}")
                
                # Print detailed classification report
                print("\nClassification Report for", name)
                print(classification_report(y_test, y_pred, 
                      target_names=['Fast (≤30)', 'Medium (31-45)', 'Slow (>45)']))
                print("-" * 50)
                
            except Exception as e:
                print(f"Error evaluating {name}: {str(e)}")
        
        # Calculate ensemble accuracy (majority voting)
        ensemble_predictions = []
        for i in range(len(y_test)):
            votes = [all_predictions[model][i] for model in all_predictions]
            ensemble_pred = max(set(votes), key=votes.count)
            ensemble_predictions.append(ensemble_pred)
        
        ensemble_accuracy = accuracy_score(y_test, ensemble_predictions)
        print(f"\nEnsemble Accuracy (Majority Voting): {ensemble_accuracy:.4f}")
        
        # Calculate weighted ensemble accuracy using our custom weights
        weights = {
            'Logistic Regression': 0.8,
            'Random Forest': 1.0,
            'XGBoost': 1.0,
            'LightGBM': 1.0,
            'SVC': 1.5,
            'KNN': 1.5,
            'Decision Tree': 1.2
        }
        
        weighted_predictions = []
        for i in range(len(y_test)):
            weighted_votes = {0: 0, 1: 0, 2: 0}  # Fast, Medium, Slow
            for model_name, predictions in all_predictions.items():
                pred = predictions[i]
                weight = weights[model_name]
                weighted_votes[pred] += weight
            weighted_pred = max(weighted_votes.items(), key=lambda x: x[1])[0]
            weighted_predictions.append(weighted_pred)
        
        weighted_accuracy = accuracy_score(y_test, weighted_predictions)
        print(f"Weighted Ensemble Accuracy: {weighted_accuracy:.4f}")
        
        # Plot accuracy comparison
        plt.figure(figsize=(12, 6))
        accuracies = list(results.values())
        models_names = list(results.keys())
        
        bars = plt.bar(models_names, accuracies)
        plt.axhline(y=ensemble_accuracy, color='r', linestyle='--', label='Simple Ensemble')
        plt.axhline(y=weighted_accuracy, color='g', linestyle='--', label='Weighted Ensemble')
        
        plt.title('Model Accuracy Comparison')
        plt.xlabel('Models')
        plt.ylabel('Accuracy Score')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom')
        
        plt.legend()
        plt.savefig('model_accuracy_comparison.png')
        plt.close()
        
        # Calculate class-wise accuracy for fast food predictions
        print("\nClass-wise Accuracy for Fast Food Predictions:")
        print("-" * 50)
        for name, predictions in all_predictions.items():
            # Get indices of actual fast delivery samples
            fast_indices = y_test == 0
            
            # Calculate accuracy for fast delivery predictions
            fast_accuracy = accuracy_score(y_test[fast_indices], predictions[fast_indices])
            print(f"{name:20} : {fast_accuracy:.4f}")
        
        return results, ensemble_accuracy, weighted_accuracy
        
    except Exception as e:
        print(f"Error in model evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    print("Evaluating model accuracy scores...")
    results, ensemble_accuracy, weighted_accuracy = evaluate_models()
    print("\nEvaluation complete! Check model_accuracy_comparison.png for visualization.") 