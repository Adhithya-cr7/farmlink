# ============================================================
# TEST SCRIPT - test_demand_forecasting.py
# ============================================================

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# ============================================================
# 1️⃣ LOAD DATA AND MODEL
# ============================================================
data = pd.read_csv("dataset.csv")

# Load model and scaler
model = load_model("crop_demand_model.h5")
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

print("✅ Model and scaler loaded successfully")

# ============================================================
# 2️⃣ USER INPUT
# ============================================================
region_input = input("Enter Region Name: ").strip()
crop_input = input("Enter Crop Name: ").strip()

# Encode categorical columns (same way as training)
le_region = LabelEncoder()
le_crop = LabelEncoder()

data['Region'] = le_region.fit_transform(data['Region'])
data['Crop'] = le_crop.fit_transform(data['Crop'])

# Check if user input is valid
if region_input not in le_region.classes_:
    print(f"❌ Region '{region_input}' not found! Available regions: {list(le_region.classes_)}")
    exit()

if crop_input not in le_crop.classes_:
    print(f"❌ Crop '{crop_input}' not found! Available crops: {list(le_crop.classes_)}")
    exit()

# ============================================================
# 3️⃣ FILTER DATA FOR SELECTED REGION & CROP
# ============================================================
region_encoded = le_region.transform([region_input])[0]
crop_encoded = le_crop.transform([crop_input])[0]

subset = data[(data['Region'] == region_encoded) & (data['Crop'] == crop_encoded)]

if subset.empty:
    print(f"❌ No data found for {region_input} - {crop_input}")
    exit()

# Create time series
subset['Date'] = pd.to_datetime(subset[['Year', 'Month']].assign(DAY=1))
subset = subset[['Date', 'Market_Demand']].set_index('Date')
subset = subset.resample('M').mean().interpolate()

print(f"\n📊 Forecasting for Region: {region_input}, Crop: {crop_input}")
print(subset.tail())

# ============================================================
# 4️⃣ PREPARE SEQUENCE FOR PREDICTION
# ============================================================
scaled = scaler.transform(subset)

TIME_STEPS = 12
def create_sequences(data, time_steps=12):
    X = []
    for i in range(len(data) - time_steps):
        X.append(data[i:i+time_steps])
    return np.array(X)

X_test = create_sequences(scaled, TIME_STEPS)
y_actual = scaled[TIME_STEPS:]

# ============================================================
# 5️⃣ PREDICT DEMAND
# ============================================================
y_pred_scaled = model.predict(X_test)
y_pred = scaler.inverse_transform(y_pred_scaled)
y_actual_rescaled = scaler.inverse_transform(y_actual)

# ============================================================
# 6️⃣ FUTURE FORECASTING (NEXT 6 MONTHS)
# ============================================================
last_seq = scaled[-TIME_STEPS:]
future_preds = []
curr_seq = last_seq.copy()

for _ in range(6):  # next 6 months
    pred = model.predict(curr_seq.reshape(1, TIME_STEPS, 1))
    future_preds.append(pred[0, 0])
    curr_seq = np.append(curr_seq[1:], pred)[-TIME_STEPS:]

future_rescaled = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))

print("\n🔮 Next 6 Months Forecasted Demand:")
for i, val in enumerate(future_rescaled.flatten(), 1):
    print(f"Month +{i}: {val:.2f}")

# ============================================================
# 7️⃣ PLOT ACTUAL VS PREDICTED
# ============================================================
plt.figure(figsize=(10,5))
plt.plot(y_actual_rescaled, label='Actual Demand', marker='o')
plt.plot(y_pred, label='Predicted Demand', linestyle='--', marker='x')
plt.title(f"Actual vs Predicted Demand ({region_input} - {crop_input})")
plt.xlabel("Time Steps (Months)")
plt.ylabel("Market Demand")
plt.legend()
plt.grid(True)
plt.show()

# ============================================================
# 8️⃣ PLOT FUTURE FORECAST
# ============================================================
plt.figure(figsize=(8,4))
plt.plot(range(1, 7), future_rescaled, marker='o', color='green')
plt.title(f"Next 6 Months Forecast ({region_input} - {crop_input})")
plt.xlabel("Months Ahead")
plt.ylabel("Predicted Market Demand")
plt.grid(True)
plt.show()
