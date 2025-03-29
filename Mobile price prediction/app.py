# Step 2: Creating a Flask application
from flask import Flask, request, render_template, jsonify
import pickle
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# Load the model and column information
try:
    with open('mobile_price_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)

    with open('model_columns.pkl', 'rb') as columns_file:
        column_info = pickle.load(columns_file)
        numerical_cols = column_info['numerical_cols']
        categorical_cols = column_info['categorical_cols']
except FileNotFoundError as e:
    print(f"Error loading model files: {e}")
    print(f"Current working directory: {os.getcwd()}")
    raise

@app.route('/')
def home():
    return render_template('index.html')  # Fixed template path

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        brand = request.form.get('brand')
        ram = float(request.form.get('ram'))
        storage = float(request.form.get('storage'))
        battery = float(request.form.get('battery'))
        processor = request.form.get('processor')
        
        # Create a dataframe with a single row for prediction
        input_data = pd.DataFrame({
            'Brand': [brand],
            'RAM (GB)': [ram],
            'Storage (GB)': [storage],
            'Battery (mAh)': [battery],
            'Processor': [processor]
        })
        
        # Print input data for debugging
        print(f"Input data for prediction: {input_data}")
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        
        # Format price with commas for thousands
        formatted_price = f"₹{prediction:,.2f}"
        
        return render_template('index.html', 
                              prediction=f"Predicted Price: {formatted_price}",
                              input_data=f"Brand: {brand}, RAM: {ram}GB, Storage: {storage}GB, Battery: {battery}mAh, Processor: {processor}")
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Prediction error: {error_details}")
        return render_template('index.html', prediction=f"Error: {str(e)}")

# Add a route for API access
@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json()
        
        # Create a dataframe with a single row for prediction
        input_data = pd.DataFrame({
            'Brand': [data.get('brand')],
            'RAM (GB)': [float(data.get('ram'))],
            'Storage (GB)': [float(data.get('storage'))],
            'Battery (mAh)': [float(data.get('battery'))],
            'Processor': [data.get('processor')]
        })
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        
        return jsonify({
            'prediction': float(prediction),
            'formatted_price': f"₹{prediction:,.2f}",
            'input': data
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    print("Starting Mobile Price Prediction App...")
    print(f"Make sure your templates folder contains index.html")
    app.run(debug=True)
