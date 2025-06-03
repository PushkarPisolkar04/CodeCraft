from flask import Flask, request, jsonify, render_template
from predict_delivery import DeliveryPredictor
import pandas as pd
import json
import traceback

app = Flask(__name__)
predictor = DeliveryPredictor()

@app.route('/')
def index():
    try:
        # Get unique cities from the dataset
        cities = sorted(pd.read_csv('data/code.csv')['Location'].str.split(',').str[-1].str.strip().unique())
        
        # Get unique cuisines
        all_cuisines = []
        cuisines_data = pd.read_csv('data/code.csv')['Cuisines'].str.split(',')
        for cuisines in cuisines_data:
            if isinstance(cuisines, list):
                all_cuisines.extend([c.strip() for c in cuisines])
        unique_cuisines = sorted(list(set(all_cuisines)))
        
        return render_template('index.html', 
                             cities=cities if len(cities) > 0 else [],
                             cuisines=unique_cuisines if len(unique_cuisines) > 0 else [])
    except Exception as e:
        print(f"Error in index route: {str(e)}")
        print(traceback.format_exc())
        return render_template('index.html', cities=[], cuisines=[])

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        required_fields = ['location', 'cuisines', 'average_cost', 'minimum_order', 
                         'rating', 'votes', 'reviews']
        
        # Check for missing fields
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({'error': f'Missing required fields: {", ".join(missing_fields)}'}), 400
        
        # Validate numeric fields
        try:
            input_data = {
                'Location': str(data['location']),
                'Cuisines': str(data['cuisines']),
                'Average_Cost': float(data['average_cost']),
                'Minimum_Order': float(data['minimum_order']),
                'Rating': float(data['rating']),
                'Votes': int(data['votes']),
                'Reviews': int(data['reviews'])
            }
        except ValueError as e:
            return jsonify({'error': f'Invalid numeric value: {str(e)}'}), 400
        
        # Validate ranges
        if not (1 <= input_data['Rating'] <= 5):
            return jsonify({'error': 'Rating must be between 1 and 5'}), 400
        if input_data['Average_Cost'] < 0 or input_data['Minimum_Order'] < 0:
            return jsonify({'error': 'Cost values cannot be negative'}), 400
        if input_data['Votes'] < 0 or input_data['Reviews'] < 0:
            return jsonify({'error': 'Votes and Reviews cannot be negative'}), 400
        
        # Make prediction
        try:
            prediction = predictor.predict(input_data)
            if not prediction or 'predicted_time' not in prediction:
                return jsonify({'error': 'Invalid prediction result'}), 500
            
            return jsonify({
                'predicted_time': prediction['predicted_time']
            })
            
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            print(traceback.format_exc())
            return jsonify({'error': 'Error making prediction'}), 500
    
    except Exception as e:
        print(f"Error in predict route: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True) 