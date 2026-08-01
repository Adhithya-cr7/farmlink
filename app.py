import os
import numpy as np
import sqlite3
import pickle
from flask import Flask, render_template, request, session, flash, redirect, url_for
from werkzeug.security import generate_password_hash, check_password_hash

from flask import Flask, render_template, request, flash
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import traceback


# ================================================================
# 🌾 Flask Configuration
# ================================================================
app = Flask(__name__)
app.secret_key = "dyuiknbvcxswe678ijc6i"

UPLOAD_FOLDER = os.path.join("static", "uploads")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ================================================================
# 🌾 SQLite3 Database
# ================================================================
DB_NAME = "users.db"

def get_db_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

# Create table if not exists
with get_db_connection() as conn:
    conn.execute('''CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL,
                        email TEXT UNIQUE NOT NULL,
                        phone TEXT,
                        password TEXT NOT NULL
                    )''')
    conn.commit()

# ================================================================
# 🌾 Load Models and Scalers
# ================================================================
crop_model = pickle.load(open('crop_new/rf_crop_model.pkl', 'rb'))
price_model = pickle.load(open('crop_new/rf_price_model.pkl', 'rb'))
mx = pickle.load(open('crop_new/minmaxscaler.pkl', 'rb'))
sc = pickle.load(open('crop_new/standscaler.pkl', 'rb'))

# Reverse crop dictionary for decoding
reverse_crop_dict = {
    1: 'rice', 2: 'maize', 3: 'jute', 4: 'cotton', 5: 'coconut',
    6: 'papaya', 7: 'orange', 8: 'apple', 9: 'muskmelon', 10: 'watermelon',
    11: 'grapes', 12: 'mango', 13: 'banana', 14: 'pomegranate', 15: 'lentil',
    16: 'blackgram', 17: 'mungbean', 18: 'mothbeans', 19: 'pigeonpeas',
    20: 'kidneybeans', 21: 'chickpea', 22: 'coffee'
}

# ================================================================
# 🌾 Routes
# ================================================================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form["name"]
        email = request.form["email"]
        phone = request.form["phone"]
        password = request.form["password"]
        hashed_password = generate_password_hash(password)

        try:
            conn = get_db_connection()
            conn.execute("INSERT INTO users (name, email, phone, password) VALUES (?, ?, ?, ?)",
                         (name, email, phone, hashed_password))
            conn.commit()
            conn.close()
            flash("Registration successful! Please login.", "success")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("Email already registered.", "danger")

    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        conn = get_db_connection()
        user = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        conn.close()

        if user and check_password_hash(user["password"], password):
            session["user"] = {
                "id": user["id"],
                "name": user["name"],
                "email": user["email"]
            }
            flash("Login successful!", "success")
            return redirect(url_for("index"))
        else:
            flash("Invalid credentials.", "danger")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out successfully.", "info")
    return redirect(url_for("index"))

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

# ================================================================
# 🌾 Crop Recommendation Route
# ================================================================
# ================================================================
# 🌾 FULL 29+ STATES DICTIONARY (No CSV needed!)
# Place this right above your crop_recommendation route
# ================================================================
state_market_data = {
    "Andhra Pradesh": [{"name": "Rice", "price": 34}, {"name": "Maize", "price": 22}, {"name": "Mango", "price": 50}, {"name": "Papaya", "price": 30}, {"name": "Cotton", "price": 60}],
    "Arunachal Pradesh": [{"name": "Rice", "price": 35}, {"name": "Maize", "price": 24}, {"name": "Orange", "price": 45}, {"name": "Apple", "price": 90}, {"name": "Banana", "price": 25}],
    "Assam": [{"name": "Jute", "price": 40}, {"name": "Rice", "price": 34}, {"name": "Tea", "price": 150}, {"name": "Banana", "price": 20}, {"name": "Papaya", "price": 28}],
    "Bihar": [{"name": "Maize", "price": 21}, {"name": "Lentil", "price": 70}, {"name": "Rice", "price": 32}, {"name": "Mango", "price": 45}, {"name": "Banana", "price": 22}],
    "Chhattisgarh": [{"name": "Rice", "price": 33}, {"name": "Maize", "price": 22}, {"name": "Chickpea", "price": 60}, {"name": "Pigeonpeas", "price": 90}, {"name": "Papaya", "price": 25}],
    "Goa": [{"name": "Coconut", "price": 45}, {"name": "Mango", "price": 60}, {"name": "Banana", "price": 30}, {"name": "Rice", "price": 36}, {"name": "Cashew", "price": 600}],
    "Gujarat": [{"name": "Cotton", "price": 62}, {"name": "Maize", "price": 23}, {"name": "Papaya", "price": 30}, {"name": "Banana", "price": 25}, {"name": "Pigeonpeas", "price": 95}],
    "Haryana": [{"name": "Rice", "price": 35}, {"name": "Cotton", "price": 60}, {"name": "Chickpea", "price": 65}, {"name": "Maize", "price": 22}, {"name": "Muskmelon", "price": 25}],
    "Himachal Pradesh": [{"name": "Apple", "price": 95}, {"name": "Maize", "price": 24}, {"name": "Rice", "price": 35}, {"name": "Kidneybeans", "price": 110}, {"name": "Grapes", "price": 85}],
    "Jharkhand": [{"name": "Rice", "price": 33}, {"name": "Maize", "price": 22}, {"name": "Pigeonpeas", "price": 90}, {"name": "Papaya", "price": 28}, {"name": "Banana", "price": 25}],
    "Karnataka": [{"name": "Coffee", "price": 250}, {"name": "Maize", "price": 22}, {"name": "Pomegranate", "price": 90}, {"name": "Papaya", "price": 30}, {"name": "Cotton", "price": 60}],
    "Kerala": [{"name": "Coconut", "price": 45}, {"name": "Coffee", "price": 250}, {"name": "Banana", "price": 30}, {"name": "Papaya", "price": 30}, {"name": "Rice", "price": 38}],
    "Madhya Pradesh": [{"name": "Chickpea", "price": 65}, {"name": "Orange", "price": 40}, {"name": "Cotton", "price": 60}, {"name": "Maize", "price": 22}, {"name": "Lentil", "price": 70}],
    "Maharashtra": [{"name": "Cotton", "price": 60}, {"name": "Pigeonpeas", "price": 95}, {"name": "Grapes", "price": 80}, {"name": "Pomegranate", "price": 90}, {"name": "Orange", "price": 40}],
    "Manipur": [{"name": "Rice", "price": 35}, {"name": "Maize", "price": 24}, {"name": "Papaya", "price": 30}, {"name": "Banana", "price": 25}, {"name": "Orange", "price": 45}],
    "Meghalaya": [{"name": "Rice", "price": 35}, {"name": "Maize", "price": 24}, {"name": "Papaya", "price": 30}, {"name": "Banana", "price": 25}, {"name": "Orange", "price": 45}],
    "Mizoram": [{"name": "Rice", "price": 35}, {"name": "Maize", "price": 24}, {"name": "Papaya", "price": 30}, {"name": "Banana", "price": 25}, {"name": "Orange", "price": 45}],
    "Nagaland": [{"name": "Rice", "price": 35}, {"name": "Maize", "price": 24}, {"name": "Papaya", "price": 30}, {"name": "Banana", "price": 25}, {"name": "Orange", "price": 45}],
    "Odisha": [{"name": "Rice", "price": 33}, {"name": "Maize", "price": 22}, {"name": "Jute", "price": 40}, {"name": "Mango", "price": 48}, {"name": "Banana", "price": 25}],
    "Punjab": [{"name": "Rice", "price": 35}, {"name": "Maize", "price": 22}, {"name": "Cotton", "price": 60}, {"name": "Chickpea", "price": 65}, {"name": "Muskmelon", "price": 25}],
    "Rajasthan": [{"name": "Mothbeans", "price": 80}, {"name": "Chickpea", "price": 65}, {"name": "Cotton", "price": 60}, {"name": "Maize", "price": 23}, {"name": "Muskmelon", "price": 25}],
    "Sikkim": [{"name": "Maize", "price": 24}, {"name": "Rice", "price": 35}, {"name": "Orange", "price": 45}, {"name": "Cardamom", "price": 800}, {"name": "Apple", "price": 90}],
    "Tamil Nadu": [{"name": "Rice", "price": 34}, {"name": "Maize", "price": 22}, {"name": "Banana", "price": 25}, {"name": "Mango", "price": 50}, {"name": "Coconut", "price": 45}],
    "Telangana": [{"name": "Rice", "price": 34}, {"name": "Cotton", "price": 60}, {"name": "Maize", "price": 22}, {"name": "Mango", "price": 50}, {"name": "Pigeonpeas", "price": 95}],
    "Tripura": [{"name": "Rice", "price": 34}, {"name": "Papaya", "price": 26}, {"name": "Banana", "price": 25}, {"name": "Jute", "price": 40}, {"name": "Mango", "price": 48}],
    "Uttar Pradesh": [{"name": "Rice", "price": 33}, {"name": "Pigeonpeas", "price": 90}, {"name": "Mango", "price": 45}, {"name": "Watermelon", "price": 15}, {"name": "Lentil", "price": 72}],
    "Uttarakhand": [{"name": "Apple", "price": 90}, {"name": "Rice", "price": 35}, {"name": "Maize", "price": 24}, {"name": "Kidneybeans", "price": 105}, {"name": "Mango", "price": 50}],
    "West Bengal": [{"name": "Rice", "price": 34}, {"name": "Jute", "price": 40}, {"name": "Mango", "price": 48}, {"name": "Coconut", "price": 44}, {"name": "Lentil", "price": 68}],
    "Andaman and Nicobar Islands": [{"name": "Coconut", "price": 45}, {"name": "Rice", "price": 38}, {"name": "Banana", "price": 30}, {"name": "Papaya", "price": 35}, {"name": "Arecanut", "price": 300}],
    "Chandigarh": [{"name": "Rice", "price": 35}, {"name": "Maize", "price": 22}, {"name": "Mango", "price": 50}, {"name": "Banana", "price": 25}, {"name": "Papaya", "price": 30}],
    "Dadra and Nagar Haveli and Daman and Diu": [{"name": "Rice", "price": 35}, {"name": "Mango", "price": 55}, {"name": "Coconut", "price": 45}, {"name": "Banana", "price": 25}, {"name": "Papaya", "price": 30}],
    "Delhi": [{"name": "Rice", "price": 35}, {"name": "Maize", "price": 22}, {"name": "Muskmelon", "price": 25}, {"name": "Watermelon", "price": 15}, {"name": "Chickpea", "price": 65}]
}

# ================================================================
# 🌾 Crop Recommendation Route
# ================================================================
@app.route("/crop_recommendation", methods=["GET", "POST"])
def crop_recommendation():
    if request.method == "POST":
        try:
            state = request.form.get("state", "").strip()

            def get_float_or_none(field):
                val = request.form.get(field, "").strip()
                return float(val) if val else None

            N = get_float_or_none("N")
            P = get_float_or_none("P")
            K = get_float_or_none("K")
            temperature = get_float_or_none("temperature")
            humidity = get_float_or_none("humidity")
            ph = get_float_or_none("ph")
            rainfall = get_float_or_none("rainfall")
            moisture = get_float_or_none("moisture")

            # Check if all soil/weather inputs are filled (ignores state)
            values_entered = all(v is not None for v in [N, P, K, temperature, humidity, ph, rainfall, moisture])

            # CONDITION 1 & 2: VALUES ENTERED (With or Without State -> Predict via ML Model)
            if values_entered:
                features = np.array([[N, P, K, temperature, humidity, ph, rainfall, moisture]])
                scaled = mx.transform(features)
                scaled = sc.transform(scaled)

                crop_pred = crop_model.predict(scaled)[0]
                price_pred = price_model.predict(scaled)[0]

                predicted_crop = reverse_crop_dict[int(crop_pred)]

                return render_template("crop_result.html",
                                       is_state_only=False,
                                       crop=predicted_crop.capitalize(),
                                       price=round(price_pred, 2))
            
            # CONDITION 3: ONLY STATE IS ENTERED (Show Top 5 from Dictionary)
            elif state and not values_entered:
                # Look up the state in our hardcoded dictionary
                if state in state_market_data:
                    top_crops_data = state_market_data[state]
                
                    return render_template("crop_result.html",
                                           is_state_only=True,
                                           state=state,
                                           top_crops=top_crops_data)
                else:
                    flash(f"No crop data found for the state: {state}", "warning")
                    return redirect(url_for("crop_recommendation"))
            
            # CONDITION 4: NOTHING OR INCOMPLETE DATA ENTERED
            else:
                flash("Please select a State OR fill in all soil & weather parameters.", "warning")
                return redirect(url_for("crop_recommendation"))

        except Exception as e:
            print("🔥 ERROR in /crop_recommendation route:")
            traceback.print_exc() 
            flash(f"Error: {str(e)}", "danger")
            return redirect(url_for("crop_recommendation"))

    return render_template("crop_form.html")

# ============================================================
# 1️⃣ LOAD DATA
# ============================================================
farmers = pd.read_csv("customer_system/farmers.csv")
transactions = pd.read_csv("customer_system/transactions.csv")  # optional
ranked_df = pd.read_csv("customer_system/matched_farmer_buyer_pairs.csv")  # optional

# ============================================================
# 2️⃣ ENCODERS & SCALER
# ============================================================
enc_crop = LabelEncoder()
enc_loc = LabelEncoder()

farmers['Crop_Code'] = enc_crop.fit_transform(farmers['Crop_Type'])
farmers['Loc_Code'] = enc_loc.fit_transform(farmers['Location'])

scaler = MinMaxScaler()
scaler.fit(farmers[['Crop_Code', 'Quantity_kg', 'Price_per_kg', 'Loc_Code']])
farmers_scaled = scaler.transform(farmers[['Crop_Code', 'Quantity_kg', 'Price_per_kg', 'Loc_Code']])


# ============================================================
# 4️⃣ CUSTOMER SUPPORT (BUYER MATCHING) ROUTE
# ============================================================
@app.route('/customer_support', methods=['GET', 'POST'])
def customer_support():
    if request.method == 'GET':
        # Simply render the form page first
        return render_template("customer.html")

    try:
        # --- Get input values from form ---
        buyer_id = request.form['buyer_id']
        preferred_crop = request.form['preferred_crop']
        quantity_needed = float(request.form['quantity_needed'])
        max_price = float(request.form['max_price'])
        location = request.form['location']

        # --- Encode categorical fields ---
        try:
            crop_code = enc_crop.transform([preferred_crop])[0]
        except:
            crop_code = max(farmers['Crop_Code']) + 1

        try:
            loc_code = enc_loc.transform([location])[0]
        except:
            loc_code = max(farmers['Loc_Code']) + 1

        # --- Prepare buyer input ---
        buyer_array = np.array([[crop_code, quantity_needed, max_price, loc_code]])
        buyer_scaled = scaler.transform(buyer_array)

        # --- Compute similarity ---
        similarity = cosine_similarity(buyer_scaled, farmers_scaled)[0]

        # --- Hybrid score (no past interactions for new buyers) ---
        interaction = np.zeros(len(farmers))
        hybrid_score = 0.6 * similarity + 0.4 * interaction

        # --- Top 5 Farmer Matches ---
        top_indices = np.argsort(hybrid_score)[::-1][:5]
        top_farmers = farmers.iloc[top_indices].copy()
        top_farmers["Match_Score"] = hybrid_score[top_indices]

        # --- Convert to list of dicts for HTML ---
        results = []
        for _, row in top_farmers.iterrows():
            results.append({
                "Farmer_ID": row["Farmer_ID"],
                "Crop": row["Crop_Type"],
                "Quantity": row["Quantity_kg"],
                "Price": row["Price_per_kg"],
                "Location": row["Location"],
                "Score": round(row["Match_Score"], 3)
            })

        return render_template("customer.html", buyer_id=buyer_id, preferred_crop=preferred_crop,
                               quantity_needed=quantity_needed, max_price=max_price,
                               location=location, results=results)

    except Exception as e:
        print("🔥 ERROR in /customer_support route:")
        traceback.print_exc()
        flash(f"Error: {str(e)}", "danger")
        return render_template("customer.html")

# ================================================================
# 🌾 Market Demand Forecast Route
# ================================================================
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import io, base64

@app.route("/market", methods=["GET", "POST"])
def market():
    if request.method == "POST":
        try:
            region_input = request.form["region"].strip()
            crop_input = request.form["crop"].strip()

            # Load dataset and model
            data = pd.read_csv("market/dataset.csv")
            model = load_model("market/crop_demand_model.h5")
            with open("market/scaler.pkl", "rb") as f:
                scaler = pickle.load(f)

            le_region = LabelEncoder()
            le_crop = LabelEncoder()
            data['Region'] = le_region.fit_transform(data['Region'])
            data['Crop'] = le_crop.fit_transform(data['Crop'])

            # Validation
            if region_input not in le_region.classes_:
                flash(f"Region '{region_input}' not found!", "danger")
                return render_template("market.html")

            if crop_input not in le_crop.classes_:
                flash(f"Crop '{crop_input}' not found!", "danger")
                return render_template("market.html")

            region_encoded = le_region.transform([region_input])[0]
            crop_encoded = le_crop.transform([crop_input])[0]

            subset = data[(data['Region'] == region_encoded) & (data['Crop'] == crop_encoded)]
            if subset.empty:
                flash(f"No data found for {region_input} - {crop_input}", "warning")
                return render_template("market.html")

            subset['Date'] = pd.to_datetime(subset[['Year', 'Month']].assign(DAY=1))
            subset = subset[['Date', 'Market_Demand']].set_index('Date')
            subset = subset.resample('M').mean().interpolate()

            scaled = scaler.transform(subset)
            TIME_STEPS = 12
            def create_sequences(data, time_steps=12):
                X = []
                for i in range(len(data) - time_steps):
                    X.append(data[i:i+time_steps])
                return np.array(X)

            X_test = create_sequences(scaled, TIME_STEPS)
            y_actual = scaled[TIME_STEPS:]

            y_pred_scaled = model.predict(X_test)
            y_pred = scaler.inverse_transform(y_pred_scaled)
            y_actual_rescaled = scaler.inverse_transform(y_actual)

            # Future 6-month forecast
            last_seq = scaled[-TIME_STEPS:]
            future_preds = []
            curr_seq = last_seq.copy()

            for _ in range(6):
                pred = model.predict(curr_seq.reshape(1, TIME_STEPS, 1))
                future_preds.append(pred[0, 0])
                curr_seq = np.append(curr_seq[1:], pred)[-TIME_STEPS:]

            future_rescaled = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))

            # Plot 1: Actual vs Predicted
            plt.figure(figsize=(8,4))
            plt.plot(y_actual_rescaled, label='Actual Demand', marker='o')
            plt.plot(y_pred, label='Predicted Demand', linestyle='--', marker='x')
            plt.title(f"Actual vs Predicted Demand ({region_input} - {crop_input})")
            plt.xlabel("Time (Months)")
            plt.ylabel("Market Demand")
            plt.legend()
            plt.tight_layout()
            img1 = io.BytesIO()
            plt.savefig(img1, format='png')
            img1.seek(0)
            graph_url1 = base64.b64encode(img1.getvalue()).decode()

            # Plot 2: Future Forecast
            plt.figure(figsize=(8,4))
            plt.plot(range(1,7), future_rescaled, marker='o', color='green')
            plt.title(f"Next 6 Months Forecast ({region_input} - {crop_input})")
            plt.xlabel("Months Ahead")
            plt.ylabel("Predicted Demand")
            plt.tight_layout()
            img2 = io.BytesIO()
            plt.savefig(img2, format='png')
            img2.seek(0)
            graph_url2 = base64.b64encode(img2.getvalue()).decode()

            return render_template(
                "market.html",
                region=region_input,
                crop=crop_input,
                forecast=[round(val[0], 2) for val in future_rescaled],
                graph1=graph_url1,
                graph2=graph_url2
            )

        except Exception as e:
            print("🔥 ERROR in /market route:")
            traceback.print_exc()
            flash(f"Error: {str(e)}", "danger")
            return render_template("market.html")

    return render_template("market.html")


# ================================================================
# 🌾 Run App
# ================================================================
if __name__ == "__main__":
    app.run(debug=True)
