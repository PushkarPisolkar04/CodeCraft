from flask import Flask, request, jsonify, render_template
from src.predict_delivery import DeliveryPredictor
import json

app = Flask(__name__)
predictor = DeliveryPredictor()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data
        input_data = {
            'Location': request.form.get('location'),
            'Cuisines': request.form.get('cuisines'),
            'Average_Cost': float(request.form.get('average_cost')),
            'Minimum_Order': float(request.form.get('minimum_order')),
            'Rating': float(request.form.get('rating')),
            'Votes': int(request.form.get('votes')),
            'Reviews': int(request.form.get('reviews'))
        }
        
        # Make prediction
        result = predictor.predict(input_data)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True) 