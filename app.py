from flask import Flask, render_template, request,jsonify
from flask_cors import CORS
import joblib
import os

model = joblib.load(os.path.join(os.path.dirname(__file__), 'model1.pkl'))
vectorizer = joblib.load(os.path.join(os.path.dirname(__file__), 'vectorizer1.pkl'))

app = Flask(__name__)
CORS(app)
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict",methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"error": "Empty message"}), 400

    message = data["message"]
    message_bow = vectorizer.transform([message])
    prediction = model.predict(message_bow)[0]

    return jsonify({"prediction": prediction})


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
