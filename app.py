from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load("model.joblib")

encoder = model.named_steps["preprocessor"].named_transformers_["cat"].named_steps["onehot"]
status_values = encoder.categories_[0]
location_values = encoder.categories_[1]

@app.route("/", methods=["GET", "POST"])
def predict():
    prediction = None
    if request.method == "POST":
        area = float(request.form["area"])
        status = request.form["status"]
        location = request.form["location"]
        bhk = int(request.form["bhk"])

        row = pd.DataFrame([[area, status, location, bhk]],
                           columns=["area", "status", "location", "bhk"])
        prediction = model.predict(row)[0]


    return render_template("index.html",
                           predicted_price=prediction,
                           status_values=status_values,
                           location_values=location_values)

if __name__ == "__main__":
    app.run(debug=True)
