# ================================================================
# 🌾 Test Script: Crop & Market Price Prediction (Random Forest)
# ================================================================

import numpy as np
import pickle

# ================================================================
# 1️⃣ Load Saved Models & Scalers
# ================================================================
crop_model = pickle.load(open('rf_crop_model.pkl', 'rb'))
price_model = pickle.load(open('rf_price_model.pkl', 'rb'))
mx = pickle.load(open('minmaxscaler.pkl', 'rb'))
sc = pickle.load(open('standscaler.pkl', 'rb'))

# Crop dictionary (for decoding)
reverse_crop_dict = {
    1: 'rice', 2: 'maize', 3: 'jute', 4: 'cotton', 5: 'coconut',
    6: 'papaya', 7: 'orange', 8: 'apple', 9: 'muskmelon', 10: 'watermelon',
    11: 'grapes', 12: 'mango', 13: 'banana', 14: 'pomegranate', 15: 'lentil',
    16: 'blackgram', 17: 'mungbean', 18: 'mothbeans', 19: 'pigeonpeas',
    20: 'kidneybeans', 21: 'chickpea', 22: 'coffee'
}

# ================================================================
# 2️⃣ Get User Input
# ================================================================
print("\n🌾 Enter the soil and weather parameters for prediction:\n")

N = float(input("Enter Nitrogen (N): "))
P = float(input("Enter Phosphorous (P): "))
K = float(input("Enter Potassium (K): "))
temperature = float(input("Enter Temperature (°C): "))
humidity = float(input("Enter Humidity (%): "))
ph = float(input("Enter pH value: "))
rainfall = float(input("Enter Rainfall (mm): "))
moisture = float(input("Enter Moisture (%): "))

# ================================================================
# 3️⃣ Preprocess Input
# ================================================================
features = np.array([[N, P, K, temperature, humidity, ph, rainfall, moisture]])
scaled = mx.transform(features)
scaled = sc.transform(scaled)

# ================================================================
# 4️⃣ Predict Crop and Market Price
# ================================================================
crop_pred = crop_model.predict(scaled)[0]
price_pred = price_model.predict(scaled)[0]

# ================================================================
# 5️⃣ Display Result
# ================================================================
predicted_crop = reverse_crop_dict[int(crop_pred)]
print("\n================ Prediction Result ================")
print(f"🌾 Recommended Crop: {predicted_crop.capitalize()}")
print(f"💰 Estimated Market Price: ₹{price_pred:.2f}")
print("===================================================")
