import os
import joblib
import numpy as np
from flask import Flask, request, jsonify

# ===================================================
# 1. SETUP AND LOAD ARTIFACTS
# ===================================================

# Initialize the Flask application
app = Flask(__name__)

# --- Define the relative paths to your model artifacts ---
# This script assumes the 'models' folder is in the same directory as this app.py file.
MODELS_DIR = "models"

# --- Load all the saved artifacts from the 'models' folder ---
try:
    print("🚀 Loading model and preprocessing artifacts...")
    model = joblib.load(os.path.join(MODELS_DIR, "final_model.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    selector = joblib.load(os.path.join(MODELS_DIR, "selector.pkl"))
    label_encoder = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))
    print("✅ Artifacts loaded successfully!")
except FileNotFoundError as e:
    print(f"❌ Error: A required model file was not found: {e}")
    print("Please make sure you have run the training script (`final.py`) first to generate the models.")
    model = None

# ===================================================
# 2. CREATE THE PREDICTION API ENDPOINT
# ===================================================

@app.route('/predict', methods=['POST'])
def predict():
    """
    Receives a malware feature vector via a POST request and returns the
    predicted malware family and confidence score.
    """
    if model is None:
        return jsonify({"error": "Model artifacts are not loaded. Server is not ready."}), 500

    data = request.get_json()
    if not data or 'features' not in data:
        return jsonify({"error": "Invalid input: Please provide a JSON object with a 'features' key."}), 400

    try:
        features = data['features']
        X_new = np.array(features).reshape(1, -1)
        X_new_scaled = scaler.transform(X_new)
        X_new_selected = selector.transform(X_new_scaled)
        
        prediction_encoded = model.predict(X_new_selected)[0]
        prediction_probabilities = model.predict_proba(X_new_selected)
        
        confidence = prediction_probabilities[0][prediction_encoded]
        family_name = label_encoder.inverse_transform([prediction_encoded])[0]
        
        response = {
            'prediction': {
                'family_name': family_name,
                'encoded_label': int(prediction_encoded),
                'confidence': float(confidence)
            }
        }
        
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": f"An error occurred during prediction: {str(e)}"}), 500

# ===================================================
# 3. RUN THE FLASK APP
# ===================================================
if __name__ == '__main__':
    # Run the app on localhost, port 5000
    app.run(port=5000, debug=True)

