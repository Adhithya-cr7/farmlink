# Farmlink: AI-Powered Agricultural Decision Support System

## Overview
Farmlink is a unified Flask-based web application designed to assist farmers, traders, and agricultural planners through data-driven, intelligent recommendations. By integrating Machine Learning and Deep Learning, this system bridges the gap between agricultural production and market demand, empowering users with actionable insights for sustainable farming. 

## Key Features
* **Crop & Price Recommendation**: Utilizes a Random Forest classifier to suggest optimal crops based on soil nutrients (Nitrogen, Phosphorus, Potassium) and environmental factors (Temperature, Humidity, pH, Rainfall, Moisture). It also estimates expected market prices. The model achieved an accuracy of approximately 97.47% during testing.
* **Buyer–Farmer Matching**: Employs cosine similarity and MinMaxScaler to connect buyers and farmers transparently based on crop type, location, price, and quantity preferences.
* **Market Demand Forecasting**: Leverages a Long Short-Term Memory (LSTM) neural network to predict future agricultural demand trends over a six-month window. The forecasting model achieved a Mean Absolute Error (MAE) of 3.7%, with insights displayed via dynamic Matplotlib graphs.
* **Secure User Management**: Features an SQLite3 database with Werkzeug password hashing to safely manage farmer and buyer sessions locally.

## Technology Stack
* **Backend Framework**: Flask 2.2+.
* **Machine Learning**: Scikit-learn 1.3+ (Random Forest Classifier & Regressor).
* **Deep Learning**: TensorFlow / Keras 2.11+ (LSTM).
* **Data Processing**: Pandas, NumPy.
* **Visualization**: Matplotlib.
* **Database**: SQLite3.

## System Architecture
The application follows a modular architecture where Flask routes user inputs from HTML forms to pre-trained `.pkl` and `.h5` model files. Results are rendered dynamically via Jinja2 templates, with all visualizations embedded directly into the web interface as base64-encoded images for a seamless real-time experience.

## Author
**Adithya Kumar Tummala**
