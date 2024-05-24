from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load("flame_model.joblib")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get user input from the form
        flame_detection = int(request.form['flame_detection'])
        humidity = float(request.form['humidity'])
        temperature = float(request.form['temperature'])
        gas_level = float(request.form['gas_level'])

        # Create a DataFrame with the input values
        input_data = pd.DataFrame({
            'Flame Detection': [flame_detection],
            'Humidity (%)': [humidity],
            'Temperature (°C)': [temperature],
            'Gas Level (PPM)': [gas_level]
        })

        # Make prediction using the model
        result = model.predict(input_data)[0]

        # Map the result to the corresponding safety condition
        safety_condition = "Safe" if result == 1 else "Unsafe"

        return render_template('result.html', safety_condition=safety_condition)

if __name__ == '__main__':
    app.run(debug=True)