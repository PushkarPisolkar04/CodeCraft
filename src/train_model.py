import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from datetime import datetime

def train_models():
    # Load the training data
    print("Loading training data...")
    train_data = pd.read_csv('data/train.csv')
    
    # Replace class names to avoid special characters
    train_data['Delivery_Category'] = train_data['Delivery_Category'].replace({
        'Fast (≤30 mins)': 'Fast (<=30 mins)',
        'Medium (31-45 mins)': 'Medium (31-45 mins)',
        'Slow (>45 mins)': 'Slow (>45 mins)'
    })
    
    # Separate features and target
    X_train = train_data.drop('Delivery_Category', axis=1)
    y_train = train_data['Delivery_Category']
    
    # Encode target variable
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    
    # Save the label encoder
    joblib.dump(le, 'models/target_encoder.joblib')
    
    # Initialize models
    models = {
        'logistic_regression': LogisticRegression(multi_class='multinomial', max_iter=1000),
        'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'xgboost': xgb.XGBClassifier(objective='multi:softmax', num_class=3),
        'lightgbm': lgb.LGBMClassifier(),
        'svc': SVC(probability=True),
        'knn': KNeighborsClassifier(n_neighbors=5),
        'decision_tree': DecisionTreeClassifier(random_state=42)
    }
    
    # Train and save each model
    trained_models = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train_encoded)
        trained_models[name] = model
        
        # Save the model
        joblib.dump(model, f'models/{name}_model.joblib')
    
    return trained_models, le

def display_evaluation_results(results):
    """Display evaluation results in a formatted way in the terminal"""
    print("\n" + "="*80)
    print("MODEL EVALUATION REPORT")
    print("Generated on:", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print("="*80)
    
    # Display summary table
    print("\nSUMMARY OF RESULTS")
    print("-"*80)
    print(f"{'Model':<20} {'Accuracy':<10} {'Macro Avg F1':<15} {'Weighted Avg F1':<15}")
    print("-"*80)
    
    for model_name, model_results in results.items():
        accuracy = model_results['classification_report']['accuracy']
        macro_f1 = model_results['classification_report']['macro avg']['f1-score']
        weighted_f1 = model_results['classification_report']['weighted avg']['f1-score']
        print(f"{model_name:<20} {accuracy:.3f}      {macro_f1:.3f}         {weighted_f1:.3f}")
    
    # Display detailed results for each model
    print("\nDETAILED RESULTS")
    print("="*80)
    
    for model_name, model_results in results.items():
        print(f"\n{model_name.upper()}")
        print("-"*40)
        
        # Display classification report
        print("\nClassification Report:")
        print("-"*40)
        print(f"{'Class':<20} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
        print("-"*60)
        
        for class_name, metrics in model_results['classification_report'].items():
            if isinstance(metrics, dict):
                print(f"{class_name:<20} {metrics['precision']:.3f}      {metrics['recall']:.3f}      "
                      f"{metrics['f1-score']:.3f}      {metrics['support']}")
        
        # Display confusion matrix
        print("\nConfusion Matrix:")
        print("-"*40)
        cm = np.array(model_results['confusion_matrix'])
        print(np.array2string(cm, separator=', '))
        print("\n")

def evaluate_models(models, label_encoder):
    # Load the test data
    print("Loading test data...")
    test_data = pd.read_csv('data/test.csv')
    
    # Replace class names to avoid special characters
    test_data['Delivery_Category'] = test_data['Delivery_Category'].replace({
        'Fast (≤30 mins)': 'Fast (<=30 mins)',
        'Medium (31-45 mins)': 'Medium (31-45 mins)',
        'Slow (>45 mins)': 'Slow (>45 mins)'
    })
    
    # Separate features and target
    X_test = test_data.drop('Delivery_Category', axis=1)
    y_test = test_data['Delivery_Category']
    
    # Encode test target
    y_test_encoded = label_encoder.transform(y_test)
    
    # Evaluate each model
    results = {}
    for name, model in models.items():
        print(f"Evaluating {name}...")
        
        # Make predictions
        y_pred_encoded = model.predict(X_test)
        
        # Convert predictions back to original labels
        y_pred = label_encoder.inverse_transform(y_pred_encoded)
        
        # Generate classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Generate confusion matrix
        cm = confusion_matrix(y_test_encoded, y_pred_encoded).tolist()
        
        # Store results
        results[name] = {
            'classification_report': report,
            'confusion_matrix': cm
        }
    
    # Display evaluation results
    display_evaluation_results(results)
    
    return results

def main():
    # Train models
    print("Training models...")
    trained_models, label_encoder = train_models()
    
    # Evaluate models
    print("\nEvaluating models...")
    evaluation_results = evaluate_models(trained_models, label_encoder)
    
    print("\nModel training and evaluation completed!")

if __name__ == "__main__":
    main() 