# Delivery Time Prediction Project

This project predicts food delivery times based on various restaurant and order features using machine learning.

## Project Structure

```
.
├── data/
│   └── delivery_data.csv
├── models/
│   ├── delivery_time_model.joblib
│   └── feature_names.joblib
├── src/
│   ├── static/
│   │   └── script.js
│   ├── templates/
│   │   └── index.html
│   ├── app.py
│   └── train_model.py
└── requirements.txt
```

## Setup Instructions

1. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Place your dataset:
   - Put your delivery dataset as `delivery_data.csv` in the `data/` directory
   - Required columns: Restaurant_ID, Location, Cuisines, Average_Cost_for_Two, Rating, Votes, Delivery_Time

## Training the Model

1. Train the model by running:
   ```bash
   python src/train_model.py
   ```
   This will:
   - Load and preprocess the data
   - Train a Random Forest model
   - Save the model and feature names to the `models/` directory
   - Print model performance metrics

## Running the Application

1. Start the Flask backend:
   ```bash
   python src/app.py
   ```
   The API will be available at `http://localhost:5000`

2. Open `src/templates/index.html` in your web browser to use the prediction interface

## API Endpoints

- `POST /predict`: Make a delivery time prediction
  - Input JSON format:
    ```json
    {
        "average_cost": 500,
        "rating": 4.5,
        "votes": 1000,
        "location": "Downtown",
        "cuisines": "Italian"
    }
    ```
  - Returns predicted delivery time in minutes

- `GET /health`: Health check endpoint

## Model Features

The model uses the following features:
- Average Cost for Two (numerical)
- Rating (numerical, 0-5)
- Votes (numerical)
- Location (categorical)
- Cuisines (categorical)

## Deployment (Optional)

To deploy the application:

1. Backend:
   - Deploy the Flask application to a cloud service (e.g., Heroku, AWS, or Render)
   - Update the API URL in `src/static/script.js` to point to your deployed backend

2. Frontend:
   - Host the static files (HTML, JS) on a static hosting service
   - Or serve them through the same backend server

## Notes

- The model uses Random Forest Regression for predictions
- Categorical variables are encoded using One-Hot Encoding
- Numerical variables are scaled using StandardScaler
- Missing values are handled by filling with median (numerical) or mode (categorical) 