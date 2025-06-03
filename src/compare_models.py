import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
import joblib
import re
import sys

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
    print("Cleaning data...")
    sys.stdout.flush()
    
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

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model performance"""
    print(f"\nEvaluating {model_name}...")
    sys.stdout.flush()
    
    y_pred = model.predict(X_test)
    mae = np.mean(np.abs(y_test - y_pred))
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    accuracy = 100 * (1 - mae / np.mean(y_test))
    
    print(f"\n{model_name} Performance Metrics:")
    print(f"- Accuracy: {accuracy:.2f}%")
    print(f"- Mean Absolute Error: {mae:.2f} minutes")
    print(f"- Root Mean Square Error: {rmse:.2f} minutes")
    sys.stdout.flush()
    
    return accuracy, mae, rmse

def main():
    # Load and clean data
    print("Loading data from code.csv...")
    sys.stdout.flush()
    
    df = pd.read_csv('data/code.csv')
    df = clean_data(df)
    
    # Prepare features and target
    X = df[['Location', 'Cuisines', 'Average_Cost_for_Two', 'Rating', 'Votes', 'Reviews']]
    y = df['Delivery_Time']
    
    # Remove rows with missing target values
    mask = ~y.isna()
    X = X[mask]
    y = y[mask]
    
    # Split the data into training and testing sets
    print("\nSplitting data into train and test sets...")
    sys.stdout.flush()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create preprocessing pipeline
    preprocessor = create_preprocessing_pipeline()
    
    # Create and train Linear Regression model
    print("\nTraining Linear Regression model...")
    sys.stdout.flush()
    
    lr_model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])
    lr_model.fit(X_train, y_train)
    
    # Create and train Gradient Boosting model
    print("\nTraining Gradient Boosting model...")
    sys.stdout.flush()
    
    gb_model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        ))
    ])
    gb_model.fit(X_train, y_train)
    
    # Evaluate both models
    lr_accuracy, lr_mae, lr_rmse = evaluate_model(lr_model, X_test, y_test, "Linear Regression")
    gb_accuracy, gb_mae, gb_rmse = evaluate_model(gb_model, X_test, y_test, "Gradient Boosting")
    
    # Select the best model
    if gb_accuracy > lr_accuracy:
        best_model = gb_model
        print("\nGradient Boosting performed better! Using it as the final model.")
    else:
        best_model = lr_model
        print("\nLinear Regression performed better! Using it as the final model.")
    sys.stdout.flush()
    
    # Save train and test datasets
    print("\nSaving train and test datasets...")
    sys.stdout.flush()
    
    train_df = pd.concat([X_train, y_train.rename('Delivery_Time')], axis=1)
    test_df = pd.concat([X_test, y_test.rename('Delivery_Time')], axis=1)
    
    train_df.to_csv('data/train.csv', index=False)
    test_df.to_csv('data/test.csv', index=False)
    
    # Save the best model and preprocessor
    print("Saving model and preprocessor...")
    sys.stdout.flush()
    
    joblib.dump(best_model, 'data/model.pkl')
    joblib.dump(preprocessor, 'data/preprocessor.pkl')
    
    # Print dataset sizes
    print("\nDataset Sizes:")
    print(f"- Total samples: {len(df)}")
    print(f"- Training samples: {len(X_train)}")
    print(f"- Testing samples: {len(X_test)}")
    sys.stdout.flush()
    
    # Make a sample prediction
    sample_restaurant = {
        'Location': 'Sector 1, Noida',
        'Cuisines': 'North Indian, Chinese',
        'Average_Cost_for_Two': 250,
        'Rating': 4.5,
        'Votes': 100,
        'Reviews': 50
    }
    
    # Convert sample to DataFrame for prediction
    sample_df = pd.DataFrame([sample_restaurant])
    predicted_time = best_model.predict(sample_df)[0]
    
    print("\nSample Prediction:")
    print("Restaurant details:")
    for key, value in sample_restaurant.items():
        print(f"- {key}: {value}")
    print(f"\nPredicted Delivery Time: {predicted_time:.2f} minutes")
    sys.stdout.flush()

if __name__ == "__main__":
    main() 