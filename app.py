from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/prediction')
def prediction_page():
    return render_template('prediction.html')

@app.route('/predict', methods=['POST'])
def predict():
    age = int(request.form['age'])
    income = int(request.form['income'])
    satisfaction = int(request.form['satisfaction'])
    years = int(request.form['years'])
    overtime = int(request.form['overtime'])

    features = np.array([[age, income, satisfaction, years, overtime]])
    prediction = model.predict(features)[0]

    result = "⚠️ High Risk of Leaving" if prediction == 1 else "✅ Low Risk"

    return render_template('prediction.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)