import pandas as pd
import numpy as np
import joblib
import json

class DeliveryPredictor:
    def __init__(self):
        try:
            # Load all trained models
            self.models = {
                'logistic_regression': joblib.load('models/logistic_regression_model.joblib'),
                'random_forest': joblib.load('models/random_forest_model.joblib'),
                'xgboost': joblib.load('models/xgboost_model.joblib'),
                'lightgbm': joblib.load('models/lightgbm_model.joblib'),
                'svc': joblib.load('models/svc_model.joblib'),
                'knn': joblib.load('models/knn_model.joblib'),
                'decision_tree': joblib.load('models/decision_tree_model.joblib')
            }
            
            # Map prediction indices to actual delivery times from data
            self.time_mapping = {
                0: 10,   # Fastest delivery
                1: 20,   # Very fast delivery
                2: 30,   # Fast delivery (most common)
                3: 45,   # Medium delivery
                4: 65,   # Slow delivery
                5: 80,   # Very slow delivery
                6: 120   # Extremely slow delivery
            }
            
            print("Models loaded successfully")
            
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            raise
    
    def preprocess_input(self, input_data):
        """Preprocess a single input for prediction."""
        try:
            # Convert input to DataFrame
            if isinstance(input_data, dict):
                input_data = pd.DataFrame([input_data])
            
            # Initialize processed data
            processed_data = pd.DataFrame()
            
            # Clean monetary values
            for col in ['Average_Cost', 'Minimum_Order']:
                input_data[col] = input_data[col].astype(str).str.replace('₹', '').str.replace(',', '').astype(float)
            
            # Process cuisines
            input_data['Cuisine_Count'] = input_data['Cuisines'].str.count(',') + 1
            input_data['Has_North_Indian'] = input_data['Cuisines'].str.contains('North Indian', case=False, na=False).astype(int)
            input_data['Has_South_Indian'] = input_data['Cuisines'].str.contains('South Indian', case=False, na=False).astype(int)
            input_data['Has_Chinese'] = input_data['Cuisines'].str.contains('Chinese', case=False, na=False).astype(int)
            input_data['Has_Fast_Food'] = input_data['Cuisines'].str.contains('Fast Food', case=False, na=False).astype(int)
            
            # Extract city and encode it
            input_data['City'] = input_data['Location'].str.split(',').str[-1].str.strip()
            input_data['City'] = input_data['City'].astype('category').cat.codes
            
            # Create price categories using fixed bins
            bins = [0, 200, 500, float('inf')]
            labels = ['Budget', 'Mid-range', 'Premium']
            input_data['Price_Category'] = pd.cut(
                input_data['Average_Cost'].astype(float),
                bins=bins,
                labels=labels,
                include_lowest=True
            )
            input_data['Price_Category'] = input_data['Price_Category'].astype('category').cat.codes
            
            # Copy numeric columns directly
            numeric_cols = ['Average_Cost', 'Minimum_Order', 'Rating', 'Votes', 'Reviews']
            for col in numeric_cols:
                processed_data[col] = input_data[col]
            
            # Copy categorical and feature columns
            feature_cols = ['City', 'Cuisine_Count', 'Has_North_Indian', 'Has_South_Indian', 
                          'Has_Chinese', 'Has_Fast_Food', 'Price_Category']
            for col in feature_cols:
                processed_data[col] = input_data[col]
            
            # Ensure all columns are float type
            processed_data = processed_data.astype(float)
            
            print("\nProcessed features:")
            print(processed_data.iloc[0].to_dict())
            
            return processed_data
            
        except Exception as e:
            print(f"Error in preprocessing: {str(e)}")
            raise
    
    def predict(self, input_data):
        """Make predictions using all models and return numerical delivery time."""
        try:
            # Preprocess input
            processed_data = self.preprocess_input(input_data)
            
            # Make predictions with each model
            predictions = []
            raw_predictions = {}
            
            for model_name, model in self.models.items():
                try:
                    # Get raw prediction index
                    pred_idx = model.predict(processed_data)[0]
                    # Convert to actual delivery time
                    pred_time = self.time_mapping.get(pred_idx, 45)  # Default to 45 if unknown
                    predictions.append(pred_time)
                    raw_predictions[model_name] = {
                        'index': int(pred_idx),
                        'time': pred_time
                    }
                except Exception as e:
                    print(f"Error in model {model_name} prediction: {str(e)}")
            
            if not predictions:
                raise ValueError("All models failed to make predictions")
            
            print("\nModel predictions:")
            for model_name, pred in raw_predictions.items():
                print(f"{model_name}: Class {pred['index']} -> {pred['time']} minutes")
            
            # Take the most common prediction (mode)
            predicted_time = max(set(predictions), key=predictions.count)
            print(f"\nFinal prediction: {predicted_time} minutes")
            
            return {
                'predicted_time': predicted_time,
                'model_predictions': raw_predictions
            }
            
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            raise

def main():
    # Example usage
    predictor = DeliveryPredictor()
    
    # Test with the example case
    test_data = {
        'Location': 'FTI College, Law College Road, Pune',
        'Cuisines': 'Fast Food, Rolls, Burger, Salad, Wraps',
        'Average_Cost': 200,
        'Minimum_Order': 50,
        'Rating': 3.5,
        'Votes': 12,
        'Reviews': 4
    }
    
    # Make prediction
    prediction = predictor.predict(test_data)
    print("\nTest case prediction:", prediction['predicted_time'], "minutes")

if __name__ == "__main__":
    main() 