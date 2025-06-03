import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
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
        # Remove ₹ symbol and commas
        value = value.replace('₹', '').replace(',', '')
        # Extract numeric part
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

def preprocess_data(df):
    # Clean monetary values
    df['Average_Cost'] = df['Average_Cost'].apply(clean_numeric_value)
    df['Minimum_Order'] = df['Minimum_Order'].apply(clean_numeric_value)
    
    # Clean and convert Delivery_Time to minutes
    df['Delivery_Time'] = df['Delivery_Time'].str.extract('(\d+)').astype(float)
    
    # Create delivery time categories
    df['Delivery_Category'] = pd.cut(df['Delivery_Time'], 
                                   bins=[0, 30, 45, float('inf')],
                                   labels=['Fast (≤30 mins)', 'Medium (31-45 mins)', 'Slow (>45 mins)'])
    
    # Handle missing values
    df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')
    df['Votes'] = pd.to_numeric(df['Votes'], errors='coerce')
    df['Reviews'] = pd.to_numeric(df['Reviews'], errors='coerce')
    
    # Fill missing values
    df['Average_Cost'].fillna(df['Average_Cost'].median(), inplace=True)
    df['Minimum_Order'].fillna(df['Minimum_Order'].median(), inplace=True)
    df['Rating'].fillna(df['Rating'].median(), inplace=True)
    df['Votes'].fillna(0, inplace=True)
    df['Reviews'].fillna(0, inplace=True)
    
    # Extract city from Location
    df['City'] = df['Location'].apply(lambda x: x.split(',')[-1].strip() if isinstance(x, str) else 'Unknown')
    
    # Process cuisines
    df['Cuisine_Count'] = df['Cuisines'].str.count(',') + 1
    df['Has_North_Indian'] = df['Cuisines'].str.contains('North Indian', case=False, na=False).astype(int)
    df['Has_South_Indian'] = df['Cuisines'].str.contains('South Indian', case=False, na=False).astype(int)
    df['Has_Chinese'] = df['Cuisines'].str.contains('Chinese', case=False, na=False).astype(int)
    df['Has_Fast_Food'] = df['Cuisines'].str.contains('Fast Food', case=False, na=False).astype(int)
    
    # Create price category
    df['Price_Category'] = pd.qcut(df['Average_Cost'], q=3, labels=['Budget', 'Mid-range', 'Premium'])
    
    # Drop original columns that won't be used in modeling
    columns_to_drop = ['Restaurant', 'Delivery_Time', 'Location', 'Cuisines']
    df = df.drop(columns_to_drop, axis=1)
    
    return df

def encode_and_scale_features(df, scaler=None, encoders=None, is_training=True):
    # Separate features that need different encoding
    categorical_cols = ['City', 'Price_Category', 'Delivery_Category']
    numerical_cols = ['Average_Cost', 'Minimum_Order', 'Rating', 'Votes', 'Reviews', 
                     'Cuisine_Count', 'Has_North_Indian', 'Has_South_Indian', 
                     'Has_Chinese', 'Has_Fast_Food']
    
    if is_training:
        # Initialize encoders and scaler
        encoders = {col: LabelEncoder() for col in categorical_cols}
        scaler = StandardScaler()
        
        # Fit and transform
        for col in categorical_cols[:-1]:  # Exclude target variable
            df[col] = encoders[col].fit_transform(df[col])
        
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
        
        return df, scaler, encoders
    else:
        # Transform using pre-fit encoders and scaler
        for col in categorical_cols[:-1]:  # Exclude target variable
            df[col] = encoders[col].transform(df[col])
        
        df[numerical_cols] = scaler.transform(df[numerical_cols])
        
        return df

def main():
    print("Loading and cleaning data...")
    # Read the data
    df = pd.read_csv('data/code.csv')
    
    # Preprocess the data
    df = preprocess_data(df)
    
    print("Splitting features and target...")
    # Split features and target
    X = df.drop('Delivery_Category', axis=1)
    y = df['Delivery_Category']
    
    print("Splitting into train and test sets...")
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Encoding and scaling features...")
    # Encode and scale features
    X_train, scaler, encoders = encode_and_scale_features(X_train, is_training=True)
    X_test = encode_and_scale_features(X_test, scaler, encoders, is_training=False)
    
    print("Saving processed data...")
    # Save the processed data
    pd.concat([X_train, y_train], axis=1).to_csv('data/train.csv', index=False)
    pd.concat([X_test, y_test], axis=1).to_csv('data/test.csv', index=False)
    
    # Save the scaler and encoders
    joblib.dump(scaler, 'models/scaler.joblib')
    joblib.dump(encoders, 'models/encoders.joblib')
    
    print("Data preprocessing completed!")

if __name__ == "__main__":
    main() 