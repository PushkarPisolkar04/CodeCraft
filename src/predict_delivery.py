import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
import joblib
import re

def clean_numeric_value(value):
    """Clean numeric values from the dataset"""
    if pd.isna(value) or value == '-' or value == 'NEW':
        return np.nan
    if isinstance(value, str):
        # Remove ₹ symbol and commas, convert to float
        value = value.replace('₹', '').replace(',', '')
        # Extract numeric part from strings like "30 minutes"
        numeric_part = re.search(r'\d+', value)
        if numeric_part:
            return float(numeric_part.group())
    try:
        return float(value)
    except (ValueError, TypeError):
        return np.nan

def clean_data(df):
    """Clean and preprocess the dataset"""
    # Clean numeric columns
    df['Average_Cost_for_Two'] = df['Average_Cost'].apply(clean_numeric_value)
    df['Minimum_Order'] = df['Minimum_Order'].apply(clean_numeric_value)
    df['Rating'] = df['Rating'].apply(clean_numeric_value)
    df['Votes'] = df['Votes'].apply(clean_numeric_value)
    df['Reviews'] = df['Reviews'].apply(clean_numeric_value)
    
    # Extract numeric delivery time
    df['Delivery_Time'] = df['Delivery_Time'].apply(lambda x: float(x.split()[0]) if isinstance(x, str) else np.nan)
    
    # Fill missing values
    df['Average_Cost_for_Two'] = df['Average_Cost_for_Two'].fillna(df['Average_Cost_for_Two'].median())
    df['Rating'] = df['Rating'].fillna(df['Rating'].median())
    df['Votes'] = df['Votes'].fillna(df['Votes'].median())
    df['Reviews'] = df['Reviews'].fillna(df['Reviews'].median())
    
    return df

def create_preprocessing_pipeline():
    """Create the preprocessing pipeline for features"""
    numeric_features = ['Average_Cost_for_Two', 'Rating', 'Votes', 'Reviews']
    categorical_features = ['Location', 'Cuisines']
    
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(drop='first', handle_unknown='ignore')
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor

def train_best_model(data_path='data/code.csv'):
    """Train the best performing model (Gradient Boosting)"""
    # Load and clean data
    df = pd.read_csv(data_path)
    df = clean_data(df)
    
    # Prepare features and target
    X = df[['Location', 'Cuisines', 'Average_Cost_for_Two', 'Rating', 'Votes', 'Reviews']]
    y = df['Delivery_Time']
    
    # Remove rows with missing target values
    mask = ~y.isna()
    X = X[mask]
    y = y[mask]
    
    # Create and train the pipeline
    preprocessor = create_preprocessing_pipeline()
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        ))
    ])
    
    # Fit the model
    model.fit(X, y)
    
    # Save the model
    joblib.dump(model, 'models/best_delivery_model.joblib')
    
    return model

def predict_delivery_time(restaurant_data, model=None):
    """
    Predict delivery time for a restaurant
    
    Parameters:
    restaurant_data: dict with keys:
        - Location: str
        - Cuisines: str
        - Average_Cost_for_Two: float (in ₹)
        - Rating: float (1-5)
        - Votes: int
        - Reviews: int
    model: pre-loaded model (optional)
    
    Returns:
    float: Predicted delivery time in minutes
    """
    if model is None:
        try:
            model = joblib.load('models/best_delivery_model.joblib')
        except:
            model = train_best_model()
    
    # Convert input data to DataFrame
    input_df = pd.DataFrame([restaurant_data])
    
    # Make prediction
    predicted_time = model.predict(input_df)[0]
    
    return round(predicted_time, 2)

if __name__ == "__main__":
    # Example usage
    sample_restaurant = {
        'Location': 'Sector 1, Noida',
        'Cuisines': 'North Indian, Chinese',
        'Average_Cost_for_Two': 250,
        'Rating': 4.5,
        'Votes': 100,
        'Reviews': 50
    }
    
    # Train and save the model
    print("\nTraining the model...")
    model = train_best_model()
    
    # Make a prediction
    predicted_time = predict_delivery_time(sample_restaurant, model)
    print(f"\nPredicted Delivery Time: {predicted_time} minutes")
    
    # Calculate and display model performance metrics
    df = pd.read_csv('data/code.csv')
    df = clean_data(df)
    
    # Prepare test data
    X_test = df[['Location', 'Cuisines', 'Average_Cost_for_Two', 'Rating', 'Votes', 'Reviews']]
    y_test = df['Delivery_Time']
    mask = ~y_test.isna()
    X_test = X_test[mask]
    y_test = y_test[mask]
    
    # Make predictions on test data
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mae = np.mean(np.abs(y_test - y_pred))
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    accuracy = 100 * (1 - mae / np.mean(y_test))
    
    print("\nModel Performance Metrics:")
    print(f"- Accuracy: {accuracy:.2f}%")
    print(f"- Mean Absolute Error: {mae:.2f} minutes")
    print(f"- Root Mean Square Error: {rmse:.2f} minutes") 