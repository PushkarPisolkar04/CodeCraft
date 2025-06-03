# Food Delivery Time Predictor

A machine learning-powered web application that predicts food delivery times based on various factors like location, cuisine type, cost, and restaurant ratings.

## 🌟 Features

- Location-based delivery time prediction
- Multiple cuisine selection (up to 5 cuisines)
- Real-time predictions using multiple ML models
- Interactive web interface
- Model accuracy comparison visualization
- Responsive design for all devices

## 🚀 Live Demo

Visit the live application: [Food Delivery Time Predictor](https://your-vercel-url.vercel.app)

## 🛠️ Tech Stack

- **Frontend**: HTML, CSS, JavaScript, Bootstrap 5
- **Backend**: Flask (Python)
- **Machine Learning**: scikit-learn, XGBoost, LightGBM
- **Deployment**: Vercel

## 📊 Machine Learning Models

The application uses multiple models for prediction:
- Random Forest
- XGBoost
- LightGBM
- Decision Tree
- K-Nearest Neighbors
- Support Vector Machine

## 🏗️ Project Structure

```
.
├── api/
│   └── index.py          # Main Flask application
├── data/
│   ├── code.csv          # Training data
│   └── test.csv          # Test data
├── models/               # Trained ML models
├── src/                  # Source code
├── static/              # Static files
├── templates/           # HTML templates
├── requirements.txt     # Python dependencies
└── vercel.json         # Vercel configuration
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Local Development

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/food-delivery-predictor.git
   cd food-delivery-predictor
   ```

2. Create and activate virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the application:
   ```bash
   python api/index.py
   ```

5. Open http://localhost:5000 in your browser

### 🌐 Deployment

The application is configured for deployment on Vercel:

1. Install Vercel CLI:
   ```bash
   npm i -g vercel
   ```

2. Deploy:
   ```bash
   vercel
   ```

## 📈 Model Performance

The application includes a visualization of model accuracy comparisons. View this in the web interface to understand how different models perform.

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset source: [mention source if applicable]
- Contributors and maintainers
- Open source community

## 📧 Contact

Your Name - your.email@example.com

Project Link: [https://github.com/your-username/food-delivery-predictor](https://github.com/your-username/food-delivery-predictor) 