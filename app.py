from flask import Flask, request, jsonify
from main import custom_pipeline, load_model, predict

app = Flask(__name__)
model = None

# Ensure this matches main.py
numeric_features = ["Overall Qual", "Gr Liv Area", "Garage Cars", 
                    "Total Bsmt SF", "Full Bath", "Year Built"]

@app.route("/train", methods=["POST"])
def train():
    global model
    data = request.json
    filepath = data.get("filepath")
    if not filepath:
        return jsonify({"error": "Missing filepath"}), 400

    model = custom_pipeline(filepath)
    return jsonify({"message": "Model trained and saved"}), 200

@app.route("/predict", methods=["POST"])
def predict_endpoint():
    global model
    if model is None:
        try:
            model = load_model()
        except:
            return jsonify({"error": "Model not trained yet"}), 500

    input_data = request.json
    try:
        result = predict(model, input_data, numeric_features)
        return jsonify({"predicted_price": float(result)}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/")
def home():
    return "🏡 House Price Prediction API"

if __name__ == "__main__":
    app.run(debug=True)
