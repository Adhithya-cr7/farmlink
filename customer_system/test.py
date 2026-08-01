# ============================================================
# SMART FARMER–BUYER MATCHING - USER INPUT TESTING
# ============================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# 1️⃣ LOAD PREVIOUS DATA
# ============================================================
farmers = pd.read_csv("farmers.csv")
transactions = pd.read_csv("transactions.csv")  # optional if you want CF
ranked_df = pd.read_csv("matched_farmer_buyer_pairs.csv")  # optional

# ============================================================
# 2️⃣ LOAD ENCODERS & SCALER (recreate same as training)
# ============================================================
enc_crop = LabelEncoder()
enc_loc = LabelEncoder()

# Fit encoders on farmer data
farmers['Crop_Code'] = enc_crop.fit_transform(farmers['Crop_Type'])
farmers['Loc_Code'] = enc_loc.fit_transform(farmers['Location'])

# Scaler (fit on farmers only for testing)
scaler = MinMaxScaler()
scaler.fit(farmers[['Crop_Code', 'Quantity_kg', 'Price_per_kg', 'Loc_Code']])
farmers_scaled = scaler.transform(farmers[['Crop_Code', 'Quantity_kg', 'Price_per_kg', 'Loc_Code']])

# ============================================================
# 3️⃣ GET USER INPUT
# ============================================================
print("Enter your buyer details:")
buyer_id = input("Buyer ID: ")
preferred_crop = input("Preferred Crop: ")
quantity_needed = float(input("Quantity Needed (kg): "))
max_price = float(input("Max Price per kg: "))
location = input("Location: ")

# ============================================================
# 4️⃣ PREPROCESS USER INPUT
# ============================================================
# Encode categorical values
try:
    crop_code = enc_crop.transform([preferred_crop])[0]
except:
    print("⚠ Crop not found in dataset. Using new code.")
    crop_code = max(farmers['Crop_Code']) + 1  # assign new code

try:
    loc_code = enc_loc.transform([location])[0]
except:
    print("⚠ Location not found in dataset. Using new code.")
    loc_code = max(farmers['Loc_Code']) + 1  # assign new code

# Create buyer array
buyer_array = np.array([[crop_code, quantity_needed, max_price, loc_code]])
buyer_scaled = scaler.transform(buyer_array)

# ============================================================
# 5️⃣ COMPUTE SIMILARITY WITH FARMERS
# ============================================================
similarity = cosine_similarity(buyer_scaled, farmers_scaled)[0]

# ============================================================
# 6️⃣ OPTIONAL: ADD SIMPLE INTERACTION SCORE FROM TRANSACTIONS
# ============================================================
# For a new buyer, there is no past transaction, so interaction=0
interaction = np.zeros(len(farmers))

# Hybrid score (content-based 60% + interaction 40%)
hybrid_score = 0.6 * similarity + 0.4 * interaction

# ============================================================
# 7️⃣ GET TOP 5 MATCHES
# ============================================================
top_indices = np.argsort(hybrid_score)[::-1][:5]

print("\n✅ Top 5 Farmer Matches for Buyer", buyer_id)
for idx in top_indices:
    farmer = farmers.iloc[idx]
    print(f"Farmer ID: {farmer['Farmer_ID']}, Crop: {farmer['Crop_Type']}, "
          f"Quantity: {farmer['Quantity_kg']}, Price: {farmer['Price_per_kg']}, "
          f"Location: {farmer['Location']}, Match Score: {round(hybrid_score[idx],3)}")
