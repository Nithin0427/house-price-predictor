import pandas as pd
import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify, render_template
import joblib
from flask_cors import CORS

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

MODEL_PATH = "house_price_model.pkl"

try:
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    model = None
    print(f"❌ Error loading model: {e}")

def format_inr(x: float) -> str:
    return f"₹{x:,.2f}"

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.route("/api/predict", methods=["POST"])
def api_predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    data = request.get_json(silent=True) or {}
    required = ["SquareFeet", "Bedrooms", "Bathrooms", "Neighborhood", "YearBuilt"]

    missing = [k for k in required if k not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    try:
        year_built = int(data["YearBuilt"])
        house_age = datetime.now().year - year_built

        df = pd.DataFrame([{
            "SquareFeet": float(data["SquareFeet"]),
            "Bedrooms": int(data["Bedrooms"]),
            "Bathrooms": float(data["Bathrooms"]),
            "Neighborhood": str(data["Neighborhood"]),
            "YearBuilt": year_built,
            "HouseAge": house_age
        }])

        pred_log = float(model.predict(df)[0])
        pred = float(np.expm1(pred_log))      # inverse log-transform

        return jsonify({"prediction": pred, "formatted": format_inr(pred)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/predict", methods=["POST"])
def form_predict():
    if model is None:
        return render_template("index.html", prediction_text="❌ Model not loaded. Train first."), 500

    try:
        form = request.form
        year_built = int(form.get("YearBuilt"))
        house_age = datetime.now().year - year_built

        df = pd.DataFrame([{
            "SquareFeet": float(form.get("SquareFeet")),
            "Bedrooms": int(form.get("Bedrooms")),
            "Bathrooms": float(form.get("Bathrooms")),
            "Neighborhood": form.get("Neighborhood"),
            "YearBuilt": year_built,
            "HouseAge": house_age
        }])

        pred_log = float(model.predict(df)[0])
        pred = float(np.expm1(pred_log))

        return render_template("index.html", prediction_text=f"🏠 Predicted Price: {format_inr(pred)}")
    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {e}")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
