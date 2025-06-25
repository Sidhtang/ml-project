from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from model import load_model
import os

app = Flask(__name__)

# Load model once when app starts
model = load_model()

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "message": "ML API is running"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Validate input
        if 'features' not in data:
            return jsonify({"error": "Missing 'features' in request"}), 400
        
        features = np.array(data['features']).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0].max()
        
        return jsonify({
            "prediction": int(prediction),
            "probability": float(probability),
            "status": "success"
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    try:
        data = request.get_json()
        
        if 'features_list' not in data:
            return jsonify({"error": "Missing 'features_list' in request"}), 400
        
        features_array = np.array(data['features_list'])
        predictions = model.predict(features_array)
        probabilities = model.predict_proba(features_array)
        
        results = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            results.append({
                "index": i,
                "prediction": int(pred),
                "probability": float(prob.max())
            })
        
        return jsonify({
            "predictions": results,
            "total_predictions": len(results),
            "status": "success"
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
