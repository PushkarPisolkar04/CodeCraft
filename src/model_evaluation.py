import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb
import json
from datetime import datetime

def load_and_preprocess_data(data_path):
    # Load the data
    df = pd.read_csv(data_path)
    
    # Handle missing values
    numeric_columns = ['Average_Cost_for_Two', 'Rating', 'Votes']
    for col in numeric_columns:
        df[col] = df[col].fillna(df[col].median())
    
    categorical_columns = ['Restaurant_ID', 'Location', 'Cuisines']
    for col in categorical_columns:
        df[col] = df[col].fillna(df[col].mode()[0])
    
    return df

def create_preprocessing_pipeline():
    numeric_features = ['Average_Cost_for_Two', 'Rating', 'Votes']
    categorical_features = ['Location', 'Cuisines']
    
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(drop='first', handle_unknown='ignore')
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    # Calculate cross-validation scores
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    # Calculate percentage accuracy (based on MAE relative to mean delivery time)
    mean_delivery_time = np.mean(y_test)
    accuracy_percentage = max(0, 100 * (1 - mae / mean_delivery_time))
    
    return {
        'model_name': model_name,
        'mae': round(mae, 2),
        'rmse': round(rmse, 2),
        'r2_score': round(r2, 4),
        'cv_mean': round(cv_mean, 4),
        'cv_std': round(cv_std, 4),
        'accuracy_percentage': round(accuracy_percentage, 2),
        'cv_scores': [round(score, 4) for score in cv_scores]
    }

def evaluate_all_models():
    # Load and prepare data
    data = load_and_preprocess_data('data/delivery_data.csv')
    X = data.drop(['Delivery_Time', 'Restaurant_ID'], axis=1)
    y = data['Delivery_Time']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create preprocessing pipeline
    preprocessor = create_preprocessing_pipeline()
    
    # Define models to evaluate
    models = [
        ('Linear Regression', Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', LinearRegression())
        ])),
        ('K-Nearest Neighbors', Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', KNeighborsRegressor(n_neighbors=5))
        ])),
        ('Support Vector Machine', Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', SVR(kernel='rbf'))
        ])),
        ('Decision Tree', Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', DecisionTreeRegressor(random_state=42))
        ])),
        ('Random Forest', Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
        ])),
        ('Gradient Boosting', Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', GradientBoostingRegressor(random_state=42))
        ])),
        ('XGBoost', Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', xgb.XGBRegressor(random_state=42))
        ])),
        ('LightGBM', Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', lgb.LGBMRegressor(random_state=42))
        ]))
    ]
    
    # Evaluate all models
    results = []
    for model_name, model in models:
        try:
            result = evaluate_model(model, X_train, X_test, y_train, y_test, model_name)
            results.append(result)
            print(f"\nEvaluated: {model_name}")
        except Exception as e:
            print(f"\nError evaluating {model_name}: {str(e)}")
    
    # Sort results by accuracy percentage
    results.sort(key=lambda x: x['accuracy_percentage'], reverse=True)
    
    # Save results to a JSON file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = f'model_evaluation_results_{timestamp}.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    # Create a markdown report
    markdown_path = f'model_evaluation_report_{timestamp}.md'
    with open(markdown_path, 'w') as f:
        f.write("# Model Evaluation Report\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Model Performance Summary\n\n")
        f.write("| Model | Accuracy (%) | MAE | RMSE | R² Score | CV Score (mean ± std) |\n")
        f.write("|-------|-------------|-----|------|----------|--------------------|\n")
        
        for result in results:
            f.write(f"| {result['model_name']} | {result['accuracy_percentage']}% | ")
            f.write(f"{result['mae']} | {result['rmse']} | {result['r2_score']} | ")
            f.write(f"{result['cv_mean']:.4f} ± {result['cv_std']:.4f} |\n")
        
        f.write("\n## Detailed Cross-validation Scores\n\n")
        for result in results:
            f.write(f"\n### {result['model_name']}\n")
            f.write("Cross-validation R² scores:\n")
            for i, score in enumerate(result['cv_scores'], 1):
                f.write(f"- Fold {i}: {score:.4f}\n")
    
    print(f"\nResults saved to {json_path} and {markdown_path}")
    return results

if __name__ == "__main__":
    evaluate_all_models() 