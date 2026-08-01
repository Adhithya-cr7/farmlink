# ============================================================
# TRAIN SCRIPT - train_demand_forecasting.py
# ============================================================

import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# 1️⃣ LOAD DATA
# ============================================================
data = pd.read_csv("dataset.csv")

print("✅ Data loaded successfully")
print(data.head())

# ============================================================
# 2️⃣ PREPROCESSING
# ============================================================
# Encode categorical features
le_region = LabelEncoder()
le_crop = LabelEncoder()

data['Region'] = le_region.fit_transform(data['Region'])
data['Crop'] = le_crop.fit_transform(data['Crop'])

# Create datetime column for time series
data['Date'] = pd.to_datetime(data[['Year', 'Month']].assign(DAY=1))
data.sort_values(['Region', 'Crop', 'Date'], inplace=True)

# Select a single crop-region combination for training
region_name = le_region.classes_[0]
crop_name = le_crop.classes_[0]

subset = data[(data['Region'] == le_region.transform([region_name])[0]) &
              (data['Crop'] == le_crop.transform([crop_name])[0])]

subset = subset[['Date', 'Market_Demand']].set_index('Date')
subset = subset.resample('M').mean().interpolate()

print(f"\nTraining for Region: {region_name}, Crop: {crop_name}")
print(subset.head())

# ============================================================
# 3️⃣ DATA SCALING
# ============================================================
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(subset)

# Create time sequences for LSTM
def create_sequences(data, time_steps=12):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i+time_steps])
        y.append(data[i+time_steps])
    return np.array(X), np.array(y)

TIME_STEPS = 12
X, y = create_sequences(scaled, TIME_STEPS)

X_train, y_train = X, y  

# ============================================================
# 4️⃣ BUILD LSTM MODEL
# ============================================================
model = Sequential([
    LSTM(64, activation='tanh', return_sequences=True, input_shape=(TIME_STEPS, 1)),
    Dropout(0.2),
    LSTM(32, activation='tanh'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# ============================================================
# 5️⃣ TRAIN MODEL
# ============================================================
es = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
model.fit(X_train, y_train, epochs=100, batch_size=8, verbose=1, callbacks=[es])

# ============================================================
# 6️⃣ SAVE MODEL & SCALER
# ============================================================
model.save("crop_demand_model.h5")

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("\n✅ Model training completed and saved as 'crop_demand_model.h5'")
print("✅ Scaler saved as 'scaler.pkl'")
