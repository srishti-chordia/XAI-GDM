from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load trained model and scaler
model = joblib.load("random_forest_gdm_model.pkl")
scaler = joblib.load("scaler.pkl")

FEATURE_NAMES = [
    "AP",
    "ICP",
    "TD",
    "Eclampsia",
    "Age",
    "BMI",
    "ALT",
    "AST",
    "GGT",
    "ALP",
    "TBA",
    "UREA",
    "CREA",
    "UA",
    "BMG",
    "A1MG",
    "CysC",
    "FPG",
]


@app.route("/", methods=["GET", "POST"])
def predict():
    prediction = None
    probability = None

    if request.method == "POST":
        try:
            # Collect inputs in correct order
            input_values = [float(request.form[f]) for f in FEATURE_NAMES]

            input_array = np.array(input_values).reshape(1, -1)

            # Apply scaling
            input_scaled = scaler.transform(input_array)

            # Predict
            result = model.predict(input_scaled)[0]
            prob = model.predict_proba(input_scaled)[0][1]

            prediction = "⚠️ High Risk of GDM" if result == 1 else "✅ Low Risk of GDM"
            probability = round(prob * 100, 2)

        except Exception as e:
            prediction = "Error in input values. Please check again."

    return render_template(
        "index.html",
        prediction=prediction,
        probability=probability,
        features=FEATURE_NAMES,
    )


if __name__ == "__main__":
    app.run(debug=True)
