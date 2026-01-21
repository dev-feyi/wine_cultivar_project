import os
import logging
from flask import Flask, render_template, request, jsonify
from model import WineCultivarModel, train_and_save_model

# 1. Configuration & Setup
class Config:
    DEBUG = True
    PORT = 5000
    HOST = '0.0.0.0'

app = Flask(__name__)
app.config.from_object(Config)

# Configure logging for better debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 2. Initialize and Bootstrap Model
predictor = WineCultivarModel()

def bootstrap_model():
    """Ensures the model is loaded or trained before the server starts."""
    if not predictor.load_model():
        logger.info("Predictive model not found. Initializing training sequence...")
        train_and_save_model()
        predictor.load_model()
    logger.info("WineCultivarModel loaded successfully.")

bootstrap_model()

# 3. Route Handlers
@app.route('/')
def index():
    """Serves the primary UI."""
    return render_template('index.html', features=predictor.feature_names)

@app.route('/api/predict', methods=['POST'])
def handle_prediction():
    """Processes incoming feature data and returns cultivar classifications."""
    try:
        input_data = request.get_json() or {}
        validated_features = {}

        # Validation Logic
        for name in predictor.feature_names:
            val = input_data.get(name)
            
            if val is None:
                raise ValueError(f"Missing required attribute: {name}")
            
            try:
                num_val = float(val)
                if num_val < 0:
                    raise ValueError(f"Attribute {name} cannot be negative.")
                validated_features[name] = num_val
            except (TypeError, ValueError):
                raise ValueError(f"Invalid numeric input for: {name}")

        # Inference
        class_idx, probs = predictor.predict(validated_features)
        
        # Format Response
        label = int(class_idx) + 1
        return jsonify({
            'status': 'success',
            'data': {
                'prediction': {
                    'id': label,
                    'name': f"Cultivar {label}",
                    'score': round(float(probs[class_idx]) * 100, 2)
                },
                'distribution': {
                    f'C{i+1}': round(float(p) * 100, 2) for i, p in enumerate(probs)
                }
            }
        })

    except ValueError as ve:
        return jsonify({'status': 'error', 'message': str(ve)}), 400
    except Exception as e:
        logger.error(f"Prediction Error: {e}")
        return jsonify({'status': 'error', 'message': "Internal server error"}), 500

@app.route('/api/samples')
def get_sample_rows():
    """Retrieves representative samples for UI testing/demo."""
    try:
        import pandas as pd
        csv_path = os.path.join(os.path.dirname(__file__), 'data', 'wine.csv')
        df = pd.read_csv(csv_path)

        # Using a dictionary comprehension for a cleaner look
        return jsonify({
            'status': 'success',
            'samples': {
                f'type_{c+1}': df[df['cultivar'] == c].iloc[0].drop('cultivar').to_dict()
                for c in [0, 1, 2]
            }
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run()