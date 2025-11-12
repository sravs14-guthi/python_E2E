from flask import Flask, render_template, request
import joblib
import json
import pandas as pd

app = Flask(__name__)

# Load model, scaler, and feature names
model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")
with open("features.json", "r") as f:
    features = json.load(f)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Convert all input values to float
        input_values = [float(x) for x in request.form.values()]

        # Match DataFrame with model's expected features
        df_input = pd.DataFrame([input_values], columns=features)

        # Scale data
        scaled_data = scaler.transform(df_input)

        # Predict probability
        prob = model.predict_proba(scaled_data)[0][1] * 100

        # Interpretation
        if prob < 20:
            result = f"💚 Customer is safe — very low chance of leaving ({prob:.2f}%)."
        elif prob < 50:
            result = f"🟡 Customer might leave — moderate churn risk ({prob:.2f}%)."
        elif prob < 75:
            result = f"🟠 Customer is likely to leave soon ({prob:.2f}%)."
        else:
            result = f"🔴 High churn risk! Customer will probably leave ({prob:.2f}%)."

        return render_template('index.html', prediction_text=result)

    except Exception as e:
        return render_template('index.html', prediction_text=f"⚠️ Error: {str(e)}")


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
