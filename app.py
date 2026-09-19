from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Load preprocessor and model
preprocessor = joblib.load("model/preprocessor.pkl")
model = joblib.load("model/churn_model.pkl")

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        df = pd.DataFrame([data])

        # Apply preprocessing
        X_processed = preprocessor.transform(df)

        # Predict churn
        prediction = model.predict(X_processed)[0]
        probability = model.predict_proba(X_processed)[0][1]

        return jsonify({
            "prediction": "Yes" if prediction == 1 else "No",
            "churn_probability": round(float(probability), 2)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
