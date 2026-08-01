# ============================================================
# SMART FARMER–BUYER MATCHING SYSTEM (No Surprise, Fixed Scaling)
# ============================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# 1️⃣ LOAD DATA
# ============================================================
farmers = pd.read_csv("farmers.csv")      # Farmer_ID, Crop_Type, Quantity_kg, Price_per_kg, Location
buyers = pd.read_csv("buyers.csv")        # Buyer_ID, Preferred_Crop, Quantity_Needed, Max_Price, Location
transactions = pd.read_csv("transactions.csv")  # Farmer_ID, Buyer_ID, Crop_Type, Quantity, Rating

print("✅ Data Loaded Successfully")
print("Farmers:", farmers.shape, "Buyers:", buyers.shape, "Transactions:", transactions.shape)

# ============================================================
# 2️⃣ CONTENT-BASED FILTERING
# ============================================================
# Encode categorical features
enc_crop = LabelEncoder()
enc_loc = LabelEncoder()

farmers['Crop_Code'] = enc_crop.fit_transform(farmers['Crop_Type'])
buyers['Crop_Code'] = enc_crop.transform(buyers['Preferred_Crop'])

farmers['Loc_Code'] = enc_loc.fit_transform(farmers['Location'])
buyers['Loc_Code'] = enc_loc.transform(buyers['Location'])

# --- Rename buyer numeric columns to match farmer columns for scaling ---
buyers_renamed = buyers.rename(columns={
    'Quantity_Needed': 'Quantity_kg',
    'Max_Price': 'Price_per_kg'
})

# --- Fit scaler on combined data to ensure same scale ---
scaler = MinMaxScaler()
combined = pd.concat([
    farmers[['Crop_Code', 'Quantity_kg', 'Price_per_kg', 'Loc_Code']],
    buyers_renamed[['Crop_Code', 'Quantity_kg', 'Price_per_kg', 'Loc_Code']]
])

scaler.fit(combined)

farmers_scaled = scaler.transform(farmers[['Crop_Code', 'Quantity_kg', 'Price_per_kg', 'Loc_Code']])
buyers_scaled = scaler.transform(buyers_renamed[['Crop_Code', 'Quantity_kg', 'Price_per_kg', 'Loc_Code']])

# Compute cosine similarity
similarity_matrix = cosine_similarity(buyers_scaled, farmers_scaled)
print("✅ Content-based similarity computed")

# ============================================================
# 3️⃣ SIMPLE COLLABORATIVE SCORE (from transactions)
# ============================================================
# Pivot transactions to get a buyer-farmer matrix
interaction_matrix = pd.pivot_table(transactions, 
                                    index='Buyer_ID', 
                                    columns='Farmer_ID', 
                                    values='Rating', 
                                    fill_value=0)

# Align buyers and farmers
buyers_list = buyers['Buyer_ID'].unique()
farmers_list = farmers['Farmer_ID'].unique()

pred_matrix = np.zeros((len(buyers_list), len(farmers_list)))
for i, b in enumerate(buyers_list):
    for j, f in enumerate(farmers_list):
        if b in interaction_matrix.index and f in interaction_matrix.columns:
            pred_matrix[i, j] = interaction_matrix.loc[b, f]

# Normalize interaction scores
if pred_matrix.max() > 0:
    pred_matrix = pred_matrix / pred_matrix.max()
print("✅ Collaborative interaction matrix generated")

# ============================================================
# 4️⃣ HYBRID SCORE (combine both)
# ============================================================
hybrid_score = 0.6 * similarity_matrix + 0.4 * pred_matrix

# ============================================================
# 5️⃣ GENERATE TOP MATCHES
# ============================================================
recommendations = []
for i, buyer in enumerate(buyers_list):
    top_indices = np.argsort(hybrid_score[i])[::-1][:5]  # top 5 farmers
    for idx in top_indices:
        recommendations.append({
            'Buyer_ID': buyer,
            'Farmer_ID': farmers_list[idx],
            'Crop': farmers.loc[farmers['Farmer_ID'] == farmers_list[idx], 'Crop_Type'].values[0],
            'Match_Score': round(hybrid_score[i, idx], 3)
        })

ranked_df = pd.DataFrame(recommendations)
print("\n✅ Top Matches Generated!")
print(ranked_df.head(10))

# ============================================================
# 6️⃣ SAVE OUTPUT
# ============================================================
ranked_df.to_csv("matched_farmer_buyer_pairs.csv", index=False)
print("\n💾 Saved output to matched_farmer_buyer_pairs.csv")
