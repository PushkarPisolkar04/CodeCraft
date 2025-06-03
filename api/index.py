import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import json

app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), '..', 'src', 'templates'), static_folder=os.path.join(os.path.dirname(__file__), '..', 'src', 'static'))

model = None
scaler = None

def clean_numeric_value(value):
    if pd.isna(value) or value == '-' or value == 'NEW':
        return np.nan
    if isinstance(value, str):
        value = value.replace('₹', '').replace(',', '')
        numeric_part = value.split()[0] if value else ''
        try:
            return float(numeric_part)
        except (ValueError, TypeError):
            return np.nan
    try:
        return float(value)
    except (ValueError, TypeError):
        return np.nan

def clean_data(df):
    df['Average_Cost'] = df['Average_Cost'].apply(clean_numeric_value)
    df['Minimum_Order'] = df['Minimum_Order'].apply(clean_numeric_value)
    df['Rating'] = df['Rating'].apply(clean_numeric_value)
    df['Votes'] = df['Votes'].apply(clean_numeric_value)
    df['Reviews'] = df['Reviews'].apply(clean_numeric_value)
    df['Delivery_Time'] = df['Delivery_Time'].apply(lambda x: float(x.split()[0]) if isinstance(x, str) else np.nan)
    df['Average_Cost'] = df['Average_Cost'].fillna(df['Average_Cost'].median())
    df['Minimum_Order'] = df['Minimum_Order'].fillna(df['Minimum_Order'].median())
    df['Rating'] = df['Rating'].fillna(df['Rating'].median())
    df['Votes'] = df['Votes'].fillna(df['Votes'].median())
    df['Reviews'] = df['Reviews'].fillna(df['Reviews'].median())
    return df

def train_model(X, y):
    global model, scaler
    numeric_features = ['Average_Cost', 'Minimum_Order', 'Rating', 'Votes', 'Reviews']
    X_numeric = X[numeric_features].copy()
    scaler = StandardScaler()
    X_numeric_scaled = scaler.fit_transform(X_numeric)
    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_numeric_scaled, y)
    return analyze_feature_importance(X, y)

def analyze_feature_importance(X, y):
    numeric_features = ['Average_Cost', 'Minimum_Order', 'Rating', 'Votes', 'Reviews']
    X_numeric = X[numeric_features].copy()
    importance = model.feature_importances_
    feature_importance = {
        'numeric': dict(zip(numeric_features, importance.tolist())),
        'correlations': X_numeric.corrwith(y).to_dict()
    }
    return feature_importance

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        input_data = np.array([[
            float(data['average_cost']),
            float(data['minimum_order']),
            float(data['rating']),
            float(data['votes']),
            float(data['reviews'])
        ]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        return jsonify({
            'success': True,
            'predicted_time': round(prediction, 1)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/')
def index():
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'code.csv')
    df = pd.read_csv(data_path)
    df = clean_data(df)
    X = df[['Average_Cost', 'Minimum_Order', 'Rating', 'Votes', 'Reviews']]
    y = df['Delivery_Time']
    mask = ~y.isna()
    X = X[mask]
    y = y[mask]
    importance_data = train_model(X, y)
    feature_ranges = {
        'average_cost': {
            'min': int(X['Average_Cost'].min()),
            'max': int(X['Average_Cost'].max()),
            'median': int(X['Average_Cost'].median())
        },
        'minimum_order': {
            'min': int(X['Minimum_Order'].min()),
            'max': int(X['Minimum_Order'].max()),
            'median': int(X['Minimum_Order'].median())
        },
        'rating': {
            'min': round(X['Rating'].min(), 1),
            'max': round(X['Rating'].max(), 1),
            'median': round(X['Rating'].median(), 1)
        },
        'votes': {
            'min': int(X['Votes'].min()),
            'max': int(X['Votes'].max()),
            'median': int(X['Votes'].median())
        },
        'reviews': {
            'min': int(X['Reviews'].min()),
            'max': int(X['Reviews'].max()),
            'median': int(X['Reviews'].median())
        }
    }
    return render_template('index.html', importance_data=json.dumps(importance_data), feature_ranges=feature_ranges)

# Vercel entry point
app = app 