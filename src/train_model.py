import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
import lightgbm as lgb
import joblib
import os

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
    # Define numeric and categorical columns
    numeric_features = ['Average_Cost_for_Two', 'Rating', 'Votes']
    categorical_features = ['Location', 'Cuisines']
    
    # Create preprocessing steps
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(drop='first', handle_unknown='ignore')
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    # Fit the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    # Perform cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    
    print(f"\n{model_name} Performance Metrics:")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R² Score: {r2:.2f}")
    print(f"Cross-validation R² scores: {cv_scores}")
    print(f"Average CV R² Score: {cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})")
    
    return {
        'model': model,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'cv_mean': cv_scores.mean(),
        'name': model_name
    }

def train_model():
    # Load and preprocess data
    data = load_and_preprocess_data('data/delivery_data.csv')
    
    # Split features and target
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
        except Exception as e:
            print(f"\nError evaluating {model_name}: {str(e)}")
    
    # Find the best model based on R² score
    best_model = max(results, key=lambda x: x['r2'])
    print(f"\nBest performing model: {best_model['name']}")
    print(f"R² Score: {best_model['r2']:.2f}")
    
    # Save the best model
    if not os.path.exists('models'):
        os.makedirs('models')
    joblib.dump(best_model['model'], 'models/delivery_time_model.joblib')
    
    # Save feature names for later use
    feature_names = {
        'numeric_features': ['Average_Cost_for_Two', 'Rating', 'Votes'],
        'categorical_features': ['Location', 'Cuisines']
    }
    joblib.dump(feature_names, 'models/feature_names.joblib')
    
    return best_model

if __name__ == "__main__":
    train_model() 