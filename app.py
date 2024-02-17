from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load("RF_calorie_burned_predicting_model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()  # Get input data from the request
        # Process data using your model
        gender = 1 if data["Gender"].lower() == "female" else 0  # Convert gender to binary
        age = float(data["Age"])
        height = float(data["Height"])
        weight = float(data["Weight"])
        duration = float(data["Duration"])
        heart_rate = float(data["Heart Rate"])
        temperature = float(data["Temperature"])

        # Create a list of features and predict the result
        features = [[gender, age, height, weight, duration, heart_rate, temperature]]
        predictions = model.predict(features)

        return jsonify({"predictions": predictions.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)