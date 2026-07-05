from flask import Flask, render_template, request
import joblib
import pandas as pd
import os
from datetime import datetime
import webbrowser
import threading

app = Flask(__name__)

# Load model and scaler
model = joblib.load("stroke_model.pkl")
scaler = joblib.load("scaler.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    gender = int(request.form["gender"])
    age = float(request.form["age"])
    hypertension = int(request.form["hypertension"])
    heart_disease = int(request.form["heart_disease"])
    ever_married = int(request.form["ever_married"])
    work_type = int(request.form["work_type"])
    residence_type = int(request.form["residence_type"])
    avg_glucose_level = float(request.form["avg_glucose_level"])
    bmi = float(request.form["bmi"])
    smoking_status = int(request.form["smoking_status"])

    # Feature order must match training
    data = [[
        gender,
        age,
        hypertension,
        heart_disease,
        ever_married,
        work_type,
        residence_type,
        avg_glucose_level,
        bmi,
        smoking_status
    ]]

    # Scale input
    data = scaler.transform(data)

    # Prediction
    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]

    # Risk Logic
    if probability < 0.30:
        risk = "LOW RISK"
        advice = "Maintain a healthy lifestyle."

    elif probability < 0.70:
        risk = "MEDIUM RISK"
        advice = "Consult a doctor and monitor your health."

    else:
        risk = "HIGH RISK"
        advice = "Immediate medical attention is recommended."

    # Save History
    patient = {
        "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Age": age,
        "Prediction": "Stroke Risk" if prediction == 1 else "No Stroke Risk",
        "Risk": risk,
        "Probability": round(probability * 100, 2)
    }
    

    file = "patient_history.csv"

    if os.path.exists(file):
        df = pd.read_csv(file)
        df = pd.concat([df, pd.DataFrame([patient])], ignore_index=True)
    else:
        df = pd.DataFrame([patient])

    df.to_csv(file, index=False)

    return render_template(
        "index.html",
        prediction=patient["Prediction"],
        risk=risk,
        probability=patient["Probability"],
        advice=advice
    )


# ==========================
# Patient History Page
# ==========================
@app.route("/history")
def history():

    if os.path.exists("patient_history.csv"):
        df = pd.read_csv("patient_history.csv")
        history = df.to_dict(orient="records")
    else:
        history = []

    return render_template("history.html", history=history)


# Auto Open Browser
def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000")


if __name__ == "__main__":
    threading.Timer(1, open_browser).start()
    app.run(debug=True)
