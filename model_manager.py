"""
Model Management and Integration for DDI Predictor
Handles loading models, comparison results, and model selection
"""
import joblib
import json
import os
import numpy as np
import pandas as pd
from pathlib import Path

def get_results_directory():
    """Get path to results directory"""
    base_dir = Path(__file__).parent.parent / "results"
    return base_dir

def load_all_models():
    """
    Load all trained models from results directory
    
    Returns:
    --------
    dict : Dictionary of model names -> model objects
    """
    results_dir = get_results_directory()
    models = {}
    
    model_files = [
        'random_forest.pkl',
        'logistic_regression.pkl',
        'svm.pkl',
        'gradient_boosting.pkl',
        'knn.pkl'
    ]
    
    for model_file in model_files:
        model_path = results_dir / model_file
        if model_path.exists():
            try:
                model_name = model_file.replace('.pkl', '').replace('_', ' ').title()
                models[model_name] = joblib.load(model_path)
            except Exception as e:
                print(f"Error loading {model_file}: {e}")
    
    return models

def load_comparison_results():
    """
    Load model comparison results from JSON
    
    Returns:
    --------
    dict : Comparison results with metrics, summary, etc.
    """
    results_dir = get_results_directory()
    results_path = results_dir / 'comparison_results.json'
    
    if results_path.exists():
        try:
            with open(results_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading comparison results: {e}")
            return None
    return None

def get_model_performance_dataframe():
    """
    Get model performance metrics as DataFrame
    
    Returns:
    --------
    pd.DataFrame : Performance metrics for all models
    """
    results = load_comparison_results()
    if results and 'results' in results:
        return pd.DataFrame(results['results'])
    return None

def get_best_model_info():
    """
    Get information about best performing model(s)
    
    Returns:
    --------
    dict : Best model info (name, scores, etc.)
    """
    results = load_comparison_results()
    if results and 'summary' in results:
        return results['summary']
    return None

def make_prediction_with_model(model_name, features):
    """
    Make prediction with specific model
    
    Parameters:
    -----------
    model_name : str
        Name of the model to use
    features : np.array
        Feature vector
        
    Returns:
    --------
    dict : Prediction results (prediction, probability)
    """
    models = load_all_models()
    
    if model_name not in models:
        return {'error': f'Model {model_name} not found'}
    
    model = models[model_name]
    
    try:
        # Prediction
        pred = model.predict(features.reshape(1, -1))[0]
        pred_int = int(pred)
        
        # Probability
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(features.reshape(1, -1))[0]
            prob = float(proba[1])  # Probability of positive class
        else:
            # For SVM or other models without predict_proba
            prob = None
        
        return {
            'prediction': pred_int,
            'probability': prob,
            'error': None
        }
    except Exception as e:
        return {'error': str(e)}

def get_all_model_predictions(features):
    """
    Get predictions from all available models
    
    Parameters:
    -----------
    features : np.array
        Feature vector
        
    Returns:
    --------
    dict : Predictions from each model
    """
    models = load_all_models()
    predictions = {}
    
    for model_name, model in models.items():
        result = make_prediction_with_model(model_name, features)
        predictions[model_name] = result
    
    return predictions

def get_model_list():
    """Get list of available models"""
    models = load_all_models()
    return list(models.keys())

def get_default_model():
    """
    Get default model (best performing)
    Falls back to Random Forest if not available
    """
    best_info = get_best_model_info()
    if best_info and 'best_model' in best_info:
        return best_info['best_model']
    return 'Random Forest'
