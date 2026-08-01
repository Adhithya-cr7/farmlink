# ================================================================
# 🌾 Crop & Market Price Dual Prediction System using Random Forest
# ================================================================

# Install dependencies (if not already installed):
# pip install scikit-learn pandas numpy

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import pickle
import warnings
warnings.filterwarnings("ignore")

# ================================================================
# 1️⃣ Load Dataset
# ================================================================
crop = pd.read_csv("Crop_recommendation_new.csv")
crop.fillna(crop.mean(numeric_only=True), inplace=True)

# ================================================================
# 2️⃣ Encode Crop Labels
# ================================================================
crop_dict = {
    'rice': 1, 'maize': 2, 'jute': 3, 'cotton': 4, 'coconut': 5,
    'papaya': 6, 'orange': 7, 'apple': 8, 'muskmelon': 9, 'watermelon': 10,
    'grapes': 11, 'mango': 12, 'banana': 13, 'pomegranate': 14, 'lentil': 15,
    'blackgram': 16, 'mungbean': 17, 'mothbeans': 18, 'pigeonpeas': 19,
    'kidneybeans': 20, 'chickpea': 21, 'coffee': 22
}
crop['label'] = crop['label'].map(crop_dict)
reverse_crop_dict = {v: k for k, v in crop_dict.items()}

# ================================================================
# 3️⃣ Split Features and Targets
# ================================================================
X = crop[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'moisture']]
y_label = crop['label']
y_price = crop['marketing price']

# ================================================================
# 4️⃣ Train-Test Split
# ================================================================
X_train, X_test, y_label_train, y_label_test = train_test_split(X, y_label, test_size=0.2, random_state=42)
_, _, y_price_train, y_price_test = train_test_split(X, y_price, test_size=0.2, random_state=42)

# ================================================================
# 5️⃣ Feature Scaling
# ================================================================
mx = MinMaxScaler()
sc = StandardScaler()

X_train_scaled = mx.fit_transform(X_train)
X_test_scaled = mx.transform(X_test)

X_train_scaled = sc.fit_transform(X_train_scaled)
X_test_scaled = sc.transform(X_test_scaled)

# ================================================================
# 6️⃣ Random Forest - Crop Classification
# ================================================================
crop_model = RandomForestClassifier(
    n_estimators=200,        # number of trees
    max_depth=None,          # allow trees to grow fully
    random_state=42,
    n_jobs=-1                # use all CPU cores
)
crop_model.fit(X_train_scaled, y_label_train)

# Evaluate classification
y_label_pred = crop_model.predict(X_test_scaled)
accuracy = accuracy_score(y_label_test, y_label_pred)
print(f"\n🌾 Crop Prediction Accuracy: {accuracy:.4f}")

# ================================================================
# 7️⃣ Random Forest - Market Price Regression
# ================================================================
price_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)
price_model.fit(X_train_scaled, y_price_train)

# Evaluate regression
y_price_pred = price_model.predict(X_test_scaled)
r2 = r2_score(y_price_test, y_price_pred)
print(f"💰 Market Price R² Score: {r2:.4f}")

# ================================================================
# 8️⃣ Save Models and Scalers
# ================================================================
pickle.dump(crop_model, open('rf_crop_model.pkl', 'wb'))
pickle.dump(price_model, open('rf_price_model.pkl', 'wb'))
pickle.dump(mx, open('minmaxscaler.pkl', 'wb'))
pickle.dump(sc, open('standscaler.pkl', 'wb'))

