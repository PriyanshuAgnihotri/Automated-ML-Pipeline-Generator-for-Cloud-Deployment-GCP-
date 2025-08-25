import pytest
import pandas as pd
import numpy as np
from src.data_processing import load_data, preprocess_data, split_data

def test_load_data(tmp_path):
    # Create a test CSV file
    test_data = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [0.1, 0.2, 0.3, 0.4, 0.5],
        'target': [0, 1, 0, 1, 0]
    })
    
    test_file = tmp_path / "test.csv"
    test_data.to_csv(test_file, index=False)
    
    loaded_data = load_data(str(test_file))
    assert loaded_data.shape == (5, 3)
    assert 'target' in loaded_data.columns

def test_split_data():
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    y = np.array([0, 1, 0, 1, 0])
    
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, test_size=0.2, val_size=0.2)
    
    assert len(X_train) == 3
    assert len(X_val) == 1
    assert len(X_test) == 1
    assert len(y_train) == 3
    assert len(y_val) == 1
    assert len(y_test) == 1