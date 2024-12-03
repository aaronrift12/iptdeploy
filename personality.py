from flask import Flask, request, jsonify
import joblib
import traceback

# Load the trained model and vectorizer
model = joblib.load("personality_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Initialize Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request
        data = request.get_json()
        
        # Check if 'text' is in the request
        if not data or 'text' not in data:
            return jsonify({"error": "Please provide text data in the request."}), 400
        
        # Extract and vectorize input text
        text = [data['text']]  # Wrap in a list for compatibility with vectorizer
        text_vectorized = vectorizer.transform(text)
        
        # Predict the label
        prediction = model.predict(text_vectorized)[0]
        
        return jsonify({"prediction": prediction}), 200

    except Exception as e:
        # Return error details if an exception occurs
        return jsonify({"error": "An error occurred.", "details": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

