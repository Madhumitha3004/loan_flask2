from flask import Blueprint, request, jsonify
import joblib
import pandas as pd

api_bp = Blueprint('api_bp', __name__)

# Load model and scaler
model = joblib.load("models/loan_model.pkl")
scaler = joblib.load("models/scaler.pkl")
training_features = model.feature_names  # Use same features as in training

@api_bp.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        input_df = pd.DataFrame([data])

        # Fill missing dummy variables with 0
        for col in training_features:
            if col not in input_df.columns:
                input_df[col] = 0

        input_df = input_df[training_features]

        # Scale input
        scaled_input = scaler.transform(input_df)

        # Predict
        pred = model.predict(scaled_input)[0]
        prob = model.predict_proba(scaled_input)[0][1]

        return jsonify({
            "prediction": int(pred),
            "probability": float(prob)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400
