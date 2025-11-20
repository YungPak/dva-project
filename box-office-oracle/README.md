Box Office Oracle - Project Setup & Run Guide

🚀 Project Status & Achievements

We have successfully integrated the ML model, API, and UI into a complete, runnable application.

Backend API (Flask):

    . Loads the pre-trained XGBoost model (final_xgb_pipeline.pkl) using joblib.
    . Implements robust data imputation (using median/mode from imdb.csv) for non-user-input fields.
    . Calculates real-time SHAP values for every prediction to power the explanation.
    . Serves a /predict endpoint on http://127.0.0.1:5000.

Frontend UI (D3.js + HTML/CSS):

    . Single-File Solution: index.html contains all HTML structure, CSS styling, and D3.js logic.
    . Interactive Prediction: User inputs (Budget, Genre, Cast, etc.) are sent to the backend.
    . Core Innovation: Renders a dynamic SHAP Waterfall Chart showing exactly how each feature pushed the revenue up or down.
    . Exploratory Viz: Includes a robust Network Graph integration that visualizes actor collaborations (loading data from imdb.csv).

📦 Setup Instructions

Follow these steps exactly to run the solution on your local machine.

Prerequisites

    . Python 3.8 or higher installed.
    . VS Code (recommended) or any terminal.

Step 1: File Setup

Ensure your project folder contains these exact files in this structure:

/box-office-oracle
├── app.py                   # The Flask API Server
├── index.html               # The User Interface
├── imdb.csv                 # The Dataset (Required for API & Graph)
├── requirements.txt         # Python dependencies
└── model_assets/            # Folder for model files
    └── final_xgb_pipeline.pkl   # The trained model file

Step 2: Install Python Environment

    1. Open your terminal/command prompt in the project folder.
    2. Create a virtual environment (optional but recommended): python -m venv venv
    3. Activate the environment:
        . Windows: .\venv\Scripts\activate
        . Mac/Linux: source venv/bin/activate
    4. Install dependencies: pip install -r requirements.txt

Step 3: Start the Backend Server

    1. In your terminal, run: python app.py
    2. Wait until you see: * Running on http://127.0.0.1:5000
    Keep this terminal open! This is the brain of the application.

Step 4: Start the Frontend (Local Server)

Because index.html loads a CSV file, you cannot just double-click it. You must serve it.

    1. Open a second terminal window.
    2. Navigate to the project folder.
    3. Run a simple Python HTTP server: python -m http.server 8000
    4. Open your web browser and go to: http://localhost:8000/index.html

🎮 How to Use the App

Predict Revenue:
    1. Enter movie details (Title, Budget, Genre, Cast, etc.) in the left panel.
    2. Click "Predict & Explain".
    3. View the predicted revenue and the SHAP Waterfall Chart explaining the result.

Explore Networks:
    1.Scroll down to "Actor Collaboration Network".
    2. Select a Country and then an Actor to visualize their co-star connections.

🛠️ Troubleshooting

"Prediction Failed" / API Error:

    - Ensure app.py is running in a separate terminal.
    - Check if imdb.csv and model_assets/final_xgb_pipeline.pkl exist.

Network Graph is Empty:

    - Ensure you are running the HTTP server (Step 4). Opening the file directly will block the CSV load due to CORS security policies
    - Check the browser console (F12) for errors.