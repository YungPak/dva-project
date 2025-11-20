import pandas as pd
import numpy as np
import joblib
import os
from category_encoders import CatBoostEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

# --- Configuration ---
DATA_PATH = "imdb.csv"
MODEL_DIR = "model_assets"

MODEL_FILENAME = "final_xgb_pipeline.pkl" 
X_TRAIN_FILENAME = "X_train_data.joblib"

# --- Define Features ---
CATEGORICAL_COLS = ["status", "original_language", "genres", "spoken_languages", 
                    "keywords", "production_companies", "directors", "writers", "cast"]
NUMERIC_COLS = ["budget", "runtime", "vote_average", "vote_count", 
                "popularity", "averageRating", "numVotes"]
TARGET_COL = "revenue"

def load_and_preprocess_data():
    """Loads, cleans, and prepares the data for training."""
    print("Loading data...")
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"Error: Dataset not found at '{DATA_PATH}'. Please ensure 'imdb.csv' is in the same directory.")
        return None, None

    # Drop non-essential columns
    df = df.drop(["id", "backdrop_path", "homepage", "tconst", "poster_path", 
                  "tagline", "original_title", "title", "adult", 
                  "production_countries", "overview", "release_date"], axis=1)

    # Convert object types to string for encoding consistency
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype("string")

    # Drop NA values 
    df = df.dropna()
    print(f"Cleaned DataFrame has {len(df)} rows.")

    X = df.drop(TARGET_COL, axis=1)
    y = df[TARGET_COL]
    
    # Split the data 
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42)
    
    return X_train, X_val, y_train, y_val

def build_and_train_pipeline(X_train, y_train):
    """Builds and trains the ColumnTransformer and XGBoost Pipeline."""
    print("Building preprocessing pipeline...")
    
    # 1. Define Preprocessing Steps
    numeric_transformer = Pipeline([
        ("scaler", StandardScaler())
    ])
    
    # Using CatBoostEncoder for SHAP compatibility
    categorical_transformer = Pipeline([
        ("catboost_enc", CatBoostEncoder(handle_unknown='value'))
    ])
    
    # 2. Combine Transformers
    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, NUMERIC_COLS),
        ("cat", categorical_transformer, CATEGORICAL_COLS)
    ], remainder='drop')
    
    # 3. Define XGBoost Model
    xgb_regressor = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1,
        tree_method="hist",
        random_state=42
    )

    # 4. Create Final Pipeline
    full_pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("model", xgb_regressor)
    ])

    print("Training XGBoost model...")
    full_pipeline.fit(X_train, y_train)
    
    return full_pipeline

def evaluate_and_save(pipeline, X_val, y_val, X_train):
    """Evaluates the model and saves the necessary assets."""
    
    # --- Evaluation ---
    preds = pipeline.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    r2 = r2_score(y_val, preds)
    print(f"\n--- Model Performance ---")
    print(f"Validated RMSE: ${rmse:,.2f}")
    print(f"Validated R²: {r2:.3f}")
    
    # --- Saving Assets ---
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        
    model_path = os.path.join(MODEL_DIR, MODEL_FILENAME)
    X_train_path = os.path.join(MODEL_DIR, X_TRAIN_FILENAME)
    
    # Save the complete pipeline (including preprocessor and model)
    print(f"\nSaving model pipeline to: {model_path}")
    joblib.dump(pipeline, model_path)
    
    # Save a sample of X_train for SHAP initialization
    print(f"Saving X_train data for SHAP to: {X_train_path}")
    joblib.dump(X_train.sample(n=100, random_state=42), X_train_path)

    print("\n--- Phase 1 Complete ---")
    print(f"Assets saved successfully in the '{MODEL_DIR}' folder.")


if __name__ == "__main__":
    X_train, X_val, y_train, y_val = load_and_preprocess_data()
    
    if X_train is not None:
        full_pipeline = build_and_train_pipeline(X_train, y_train)
        evaluate_and_save(full_pipeline, X_val, y_val, X_train)