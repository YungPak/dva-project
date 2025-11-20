import pandas as pd
import numpy as np
import joblib
import shap
import os
from flask import Flask, request, jsonify
from flask_cors import CORS

# --- Configuration ---
MODEL_DIR = "model_assets"
MODEL_FILENAME = "final_xgb_pipeline.pkl"
DATA_PATH = "imdb.csv"

SCALE_FACTOR = 1_000_000 

# --- Define Features ---
CATEGORICAL_COLS = ["status", "original_language", "genres", "spoken_languages", 
                    "keywords", "production_companies", "directors", "writers", "cast"]
NUMERIC_COLS = ["budget", "runtime", "vote_average", "vote_count", 
                "popularity", "averageRating", "numVotes"]
TARGET_COL = "revenue"

# Fields to impute from dataset
IMPUTE_FIELDS = ['vote_average', 'vote_count', 'popularity', 'averageRating', 'numVotes']

# --- Global Variables ---
app = Flask(__name__)
CORS(app) 
full_pipeline = None
default_values = {}
X_train_sample = None
explainer = None

def load_model_and_data():
    """Loads the model, SHAP data, and calculates default imputation values."""
    global full_pipeline, default_values, X_train_sample, explainer

    try:
        model_path = os.path.join(MODEL_DIR, MODEL_FILENAME)
        
        print(f"Loading ML pipeline from {model_path}...")
        full_pipeline = joblib.load(model_path)
        
        print(f"Loading data from {DATA_PATH} for defaults and SHAP background...")
        if not os.path.exists(DATA_PATH):
             print("WARNING: imdb.csv not found. Using static defaults.")
             default_values.update({
                 'vote_average': 6.0, 'vote_count': 1000, 'popularity': 10.0,
                 'averageRating': 6.0, 'numVotes': 1000, 'runtime': 90
             })
        else:
            df = pd.read_csv(DATA_PATH).dropna() 
            
            for col in IMPUTE_FIELDS:
                if col in df.columns:
                    default_values[col] = df[col].median()
            
            if 'runtime' in df.columns:
                default_values['runtime'] = df['runtime'].median()
            else:
                default_values['runtime'] = 90.0

            # Create X_train sample 
            X_train = df.drop(TARGET_COL, axis=1)
            
            # --- SHAP INITIALIZATION ---
            preprocessor = full_pipeline.named_steps["preprocess"]
            booster = full_pipeline.named_steps["model"]
            
            # Use a larger sample (200) to better represent data distribution
            X_train_sample = X_train.sample(n=200, random_state=42)
            X_train_processed = preprocessor.transform(X_train_sample)
            
            # Use "interventional" perturbation for robustness
            explainer = shap.TreeExplainer(
                booster, 
                X_train_processed, 
                feature_perturbation="interventional", 
                model_output="raw"
            )
            
        print("Initialization complete. SHAP Explainer is ready.")
        
    except Exception as e:
        print(f"Error during initialization: {e}")
        raise

def create_input_dataframe(data):
    """
    Takes user input (JSON dict) and creates a standardized Pandas DataFrame row.
    """
    input_dict = default_values.copy()

    static_defaults = {
        'status': 'Released', 
        'original_language': 'en', 
        'spoken_languages': 'English', 
        'keywords': 'none', 
        'production_companies': 'none', 
        'writers': 'none'
    }
    
    for k, v in static_defaults.items():
        if k not in input_dict:
            input_dict[k] = v

    try:
        input_dict['budget'] = float(data.get('budget', 0)) * SCALE_FACTOR
        
        user_runtime = data.get('runtime')
        if user_runtime:
            input_dict['runtime'] = float(user_runtime)
        elif 'runtime' not in input_dict:
             input_dict['runtime'] = 90.0
             
        input_dict['genres'] = data.get('genre', 'Unknown')
        input_dict['directors'] = data.get('director', 'Unknown')
        input_dict['cast'] = data.get('cast', 'Unknown')
        
    except Exception as e:
        print(f"Error mapping user input: {e}")
        raise

    input_df = pd.DataFrame([input_dict])
    
    for col in NUMERIC_COLS:
        if col not in input_df.columns:
            input_df[col] = default_values.get(col, 0.0)
        input_df[col] = pd.to_numeric(input_df[col], errors='coerce').fillna(0.0)
            
    for col in CATEGORICAL_COLS:
        if col not in input_df.columns:
            input_df[col] = "Unknown"
        input_df[col] = input_df[col].astype(str)

    expected_cols = NUMERIC_COLS + CATEGORICAL_COLS
    return input_df[expected_feature_order_safe(input_df, expected_cols)]

def expected_feature_order_safe(df, expected_cols):
    """Helper to return valid columns that exist in the df."""
    return [c for c in expected_cols if c in df.columns]

def calculate_shap_values(input_df):
    """Calculates prediction, base value, and feature contributions."""
    
    raw_prediction = full_pipeline.predict(input_df)[0]
    
    preprocessor = full_pipeline.named_steps["preprocess"]
    input_processed = preprocessor.transform(input_df)
    
    # Calculate SHAP values
    shap_values_raw = explainer.shap_values(input_processed)
    base_value_raw = explainer.expected_value
    
    feature_names = preprocessor.get_feature_names_out()
    
    contributions = []
    
    values = shap_values_raw
    if isinstance(values, list):
        values = values[0]
    if len(values.shape) > 1:
        values = values[0]
            
    for name, impact in zip(feature_names, values):
        # Scale down the noise threshold
        if abs(impact) > 1000: 
            clean_name = name
            if '__' in name:
                parts = name.split('__')
                if len(parts) > 1:
                    clean_name = parts[1].replace('genres_', 'Genre: ').replace('cast_', 'Cast: ')
                
            contributions.append({
                "feature": clean_name,
                "impact": float(impact / SCALE_FACTOR)
            })
            
    contributions.sort(key=lambda x: abs(x['impact']), reverse=True)
    
    result = {
        "prediction": float(raw_prediction / SCALE_FACTOR),
        "base_value": float(base_value_raw / SCALE_FACTOR),
        "feature_contributions": contributions[:10],
        "message": "Success"
    }
    return result

@app.route('/predict', methods=['POST'])
def predict():
    if full_pipeline is None:
        return jsonify({"error": "Model not loaded."}), 500
        
    try:
        data = request.json
        if 'runtime' not in data:
             data['runtime'] = 90 
             
        input_df = create_input_dataframe(data)
        result = calculate_shap_values(input_df)
        return jsonify(result)
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 400

if __name__ == '__main__':
    try:
        load_model_and_data() 
        print("\n--- Starting Flask API Server on http://127.0.0.1:5000 ---")
        app.run(debug=True, port=5000, threaded=False) 
    except Exception as e:
        print(f"Failed to start server: {e}")