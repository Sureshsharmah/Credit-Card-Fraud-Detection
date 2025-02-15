from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load(r'D:\Projects\ML Projects\Credit Card Fraud Detection\credit_card_model.pkl')

# Home route (renders the UI)
@app.route('/')
def home():
    return render_template('index.html')  # Loads the HTML UI

# Prediction route (handles user input)
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        features = [float(request.form[f'V{i}']) for i in range(1, 29)]  # V1 to V28
        amount = float(request.form['Amount'])  # Amount column

        # Convert to numpy array
        input_data = np.array([features + [amount]])  # Ensure input shape matches training data

        # Make prediction
        prediction = model.predict(input_data)

        # Return the result
        result = "Fraud" if prediction[0] == 1 else "Not Fraud"
        return jsonify({'prediction': result})

    except Exception as e:
        return jsonify({'error': str(e)})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
