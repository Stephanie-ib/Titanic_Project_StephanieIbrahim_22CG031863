from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load the trained model and scaler
model_path = os.path.join('model', 'titanic_survival_model.pkl')

print("Loading model...")
with open(model_path, 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
scaler = model_data['scaler']
print("✓ Model loaded successfully!")

@app.route('/')
def home():
    """Render the home page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        # Get form data
        pclass = int(request.form['pclass'])
        sex = int(request.form['sex'])
        age = float(request.form['age'])
        fare = float(request.form['fare'])
        embarked = int(request.form['embarked'])
        
        # Prepare features for prediction
        features = np.array([[pclass, sex, age, fare, embarked]])
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]
        
        # Prepare result
        if prediction == 1:
            result = "✅ SURVIVED"
            confidence = probability[1] * 100
        else:
            result = "❌ DID NOT SURVIVE"
            confidence = probability[0] * 100
        
        return render_template('index.html', 
                             prediction_text=result,
                             confidence=f"{confidence:.2f}%")
    
    except Exception as e:
        return render_template('index.html', 
                             prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    # Use PORT environment variable for Render deployment
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=False)
    
