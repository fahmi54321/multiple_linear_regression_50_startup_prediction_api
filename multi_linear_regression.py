from flask import Flask, request, jsonify
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import wget

app = Flask(__name__)

# ===============================
# TRAIN MODEL (ON STARTUP)
# ===============================


url = "https://raw.githubusercontent.com/fahmi54321/multiple_linear_regression_50_startup_prediction_api/refs/heads/main/50_Startups.csv"
wget.download(url, '50_Startups.csv')

dataset = pd.read_csv("50_Startups.csv")

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), [3])],
    remainder='passthrough'
)

X = np.array(ct.fit_transform(X))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

# ===============================
# HELPER
# ===============================

def analyze_profit(profit):
    if profit >= 150000:
        return {
            "classification": "High Potential",
            "description": "Startup ini menunjukkan potensi profit yang tinggi berdasarkan data historis.",
            "recommendation": [
                "Pertahankan investasi R&D",
                "Optimalkan efisiensi marketing",
                "Layak untuk scaling bisnis"
            ]
        }
    elif profit >= 100000:
        return {
            "classification": "Medium Potential",
            "description": "Startup cukup stabil namun masih bisa ditingkatkan.",
            "recommendation": [
                "Evaluasi strategi marketing",
                "Optimalkan biaya operasional",
                "Perkuat diferensiasi produk"
            ]
        }
    else:
        return {
            "classification": "Low Potential",
            "description": "Profit diprediksi masih rendah dibandingkan startup lain.",
            "recommendation": [
                "Tingkatkan investasi R&D",
                "Review ulang model bisnis",
                "Kurangi biaya yang tidak berdampak"
            ]
        }

# ===============================
# API ENDPOINT
# ===============================

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    required = ["rd_spend", "administration", "marketing_spend", "state"]
    for field in required:
        if field not in data:
            return jsonify({"error": f"Missing field: {field}"}), 400

    # ===============================
    # BUILD INPUT (MATCH TRAINING)
    # ===============================
    input_data = np.array([[
        data["rd_spend"],
        data["administration"],
        data["marketing_spend"],
        data["state"]
    ]], dtype=object)

    input_transformed = ct.transform(input_data)

    predicted_profit = regressor.predict(input_transformed)[0]

    insight = analyze_profit(predicted_profit)

    return jsonify({
        "predicted_profit": round(predicted_profit, 2),
        "currency": "USD",
        "classification": insight["classification"],
        "description": insight["description"],
        "recommendation": insight["recommendation"],
        "input": data
    })

# ===============================
# RUN
# ===============================

if __name__ == "__main__":
    app.run(debug=True)
