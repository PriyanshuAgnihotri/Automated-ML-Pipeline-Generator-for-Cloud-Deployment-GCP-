import pytest
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from src.model_validation import validate_model

def test_validate_model():
    # Create test data
    X, y = make_classification(n_samples=100, n_features=4, random_state=42)
    
    # Train a simple model
    model = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
    model.fit(X, y)
    
    # Test validation
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    thresholds = {'accuracy': 0.7, 'f1_score': 0.7}
    
    validation_passed, results = validate_model(model, X, y, metrics, thresholds)
    
    assert isinstance(validation_passed, bool)
    assert all(metric in results for metric in metrics)
    assert all(0 <= results[metric] <= 1 for metric in metrics)