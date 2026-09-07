from flask import Flask, render_template, request
import joblib

from preprocess import preprocess

app = Flask(__name__)

model = joblib.load("model/spam_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    confidence = None
    message = ""

    if request.method == "POST":

        message = request.form["message"]

        processed = preprocess(message)

        vector = vectorizer.transform([processed])

        result = model.predict(vector)[0]

        probability = model.predict_proba(vector)[0]

        confidence = round(max(probability) * 100, 2)

        if result == "spam":
            prediction = "🚫 SPAM"
        else:
            prediction = "✅ HAM"

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        message=message
    )


if __name__ == "__main__":
    app.run(debug=True)