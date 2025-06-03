import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns
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

def analyze_correlations(df):
    """Analyze correlations between numeric features and target"""
    print("\nAnalyzing correlations with delivery time...")
    
    numeric_features = ['Average_Cost_for_Two', 'Rating', 'Votes', 'Reviews']
    correlations = df[numeric_features + ['Delivery_Time']].corr()['Delivery_Time'].sort_values(ascending=False)
    
    print("\nCorrelations with Delivery Time:")
    for feature, corr in correlations.items():
        if feature == 'Delivery_Time':
            print(f"- {feature}: {corr:.3f}")
    
    # Create correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[numeric_features + ['Delivery_Time']].corr(), annot=True, cmap='coolwarm')
    plt.title('Feature Correlations Heatmap')
    plt.tight_layout()
    plt.savefig('data/correlation_heatmap.png')
    plt.close()

def analyze_categorical_impact(df):
    """Analyze impact of categorical features on delivery time"""
    print("\nAnalyzing categorical features impact...")
    
    # Location analysis
    location_stats = df.groupby('Location')['Delivery_Time'].agg(['mean', 'count']).sort_values('mean', ascending=False)
    location_stats = location_stats[location_stats['count'] >= 5]  # Filter locations with at least 5 orders
    
    print("\nTop 5 Locations by Average Delivery Time:")
    print(location_stats.head().to_string())
    
    # Cuisine analysis
    df['Cuisine_List'] = df['Cuisines'].str.split(', ')
    all_cuisines = []
    for cuisines in df['Cuisine_List'].dropna():
        all_cuisines.extend(cuisines)
    
    cuisine_stats = pd.DataFrame({
        'cuisine': list(set(all_cuisines)),
        'avg_time': [df[df['Cuisines'].str.contains(cuisine, na=False)]['Delivery_Time'].mean() 
                    for cuisine in set(all_cuisines)],
        'count': [df[df['Cuisines'].str.contains(cuisine, na=False)].shape[0] 
                 for cuisine in set(all_cuisines)]
    })
    cuisine_stats = cuisine_stats[cuisine_stats['count'] >= 5].sort_values('avg_time', ascending=False)
    
    print("\nTop 5 Cuisines by Average Delivery Time:")
    print(cuisine_stats.head().to_string())

def analyze_feature_importance_gb(X, y):
    """Analyze feature importance using Gradient Boosting"""
    print("\nAnalyzing feature importance using Gradient Boosting...")
    
    # Create preprocessing pipeline
    numeric_features = ['Average_Cost_for_Two', 'Rating', 'Votes', 'Reviews']
    categorical_features = ['Location', 'Cuisines']
    
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(drop='first', handle_unknown='ignore')
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Create and train the model
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', GradientBoostingRegressor(random_state=42))
    ])
    
    model.fit(X, y)
    
    # Get feature names after preprocessing
    numeric_features_final = numeric_features
    cat_features = []
    for feature in categorical_features:
        unique_values = X[feature].unique()
        cat_features.extend([f"{feature}_{val}" for val in unique_values[1:]])
    
    all_features = numeric_features_final + cat_features
    
    # Get feature importance scores
    feature_importance = model.named_steps['regressor'].feature_importances_
    
    # Create feature importance DataFrame
    importance_df = pd.DataFrame({
        'feature': all_features[:len(feature_importance)],
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(importance_df.head(10).to_string())
    
    # Plot feature importance
    plt.figure(figsize=(12, 6))
    sns.barplot(data=importance_df.head(10), x='importance', y='feature')
    plt.title('Top 10 Most Important Features')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig('data/feature_importance.png')
    plt.close()
    
    return importance_df

def main():
    # Load and clean data
    print("Loading data from code.csv...")
    df = pd.read_csv('data/code.csv')
    df = clean_data(df)
    
    # Analyze correlations
    analyze_correlations(df)
    
    # Analyze categorical features
    analyze_categorical_impact(df)
    
    # Prepare features and target for importance analysis
    X = df[['Location', 'Cuisines', 'Average_Cost_for_Two', 'Rating', 'Votes', 'Reviews']]
    y = df['Delivery_Time']
    
    # Remove rows with missing target values
    mask = ~y.isna()
    X = X[mask]
    y = y[mask]
    
    # Analyze feature importance
    importance_df = analyze_feature_importance_gb(X, y)
    
    # Save feature importance results
    importance_df.to_csv('data/feature_importance.csv', index=False)
    
    print("\nFeature analysis completed! Results have been saved to:")
    print("- data/correlation_heatmap.png")
    print("- data/feature_importance.png")
    print("- data/feature_importance.csv")

if __name__ == "__main__":
    main() 