from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)

# Load model + scaler
model = pickle.load(open("model/model.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])

def predict():
    try:
        values = [
    request.form["Sex"],
    request.form["Age"],
    request.form["Cigs"],
    request.form["BPMeds"],
    request.form["Stroke"],
    request.form["Hyp"],
    request.form["Diabetes"],
    request.form["Chol"],
    request.form["SysBP"],
    request.form["DiaBP"],
    request.form["BMI"],
    request.form["HR"],
    request.form["Glucose"]
]
        values = [float(x) for x in request.form.values()]
        data = np.array(values).reshape(1, -1)

        data_scaled = scaler.transform(data)
        result = model.predict(data_scaled)[0]

        if result == 1:
            output = "High risk of heart disease"
        else:
            output = "Low risk"

        return render_template("index.html", prediction=output)

    except Exception as e:
        return str(e)

if __name__ == "__main__":
    app.run(debug=True)