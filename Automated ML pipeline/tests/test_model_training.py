import pytest
import numpy as np
from sklearn.datasets import make_classification
from src.model_training import train_model, evaluate_model

def test_train_model():
    # Create test data
    X, y = make_classification(n_samples=100, n_features=4, random_state=42)
    
    # Mock model config
    model_config = {
        'models': {
            'sklearn': {
                'package': 'sklearn.ensemble',
                'class': 'RandomForestClassifier',
                'required_parameters': ['n_estimators', 'max_depth']
            }
        }
    }
    
    # Test training
    model = train_model(
        X, y,
        'sklearn',
        {'n_estimators': 10, 'max_depth': 3},
        model_config
    )
    
    assert model is not None
    assert hasattr(model, 'predict')

def test_evaluate_model():
    # Create test data
    X, y = make_classification(n_samples=100, n_features=4, random_state=42)
    
    # Mock model config
    model_config = {
        'models': {
            'sklearn': {
                'package': 'sklearn.ensemble',
                'class': 'RandomForestClassifier',
                'required_parameters': ['n_estimators', 'max_depth']
            }
        }
    }
    
    # Train model
    model = train_model(
        X, y,
        'sklearn',
        {'n_estimators': 10, 'max_depth': 3},
        model_config
    )
    
    # Evaluate model
    accuracy, predictions = evaluate_model(model, X, y)
    
    assert accuracy >= 0
    assert accuracy <= 1
    assert len(predictions) == len(y)