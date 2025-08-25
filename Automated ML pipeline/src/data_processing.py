import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import logging

def load_data(file_path):
    """Load data from CSV file"""
    try:
        data = pd.read_csv(file_path)
        logging.info(f"Data loaded successfully from {file_path}")
        return data
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def preprocess_data(data, config):
    """Preprocess data based on configuration"""
    try:
        # Separate features and target
        X = data.drop('target', axis=1)
        y = data['target']
        
        # Identify numerical and categorical columns
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = X.select_dtypes(include=['object']).columns
        
        # Create preprocessing pipelines
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy=config['data_processing']['preprocessing_steps']['handle_missing_values'])),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        # Combine preprocessing steps
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ])
        
        # Apply preprocessing
        X_processed = preprocessor.fit_transform(X)
        
        logging.info("Data preprocessing completed successfully")
        return X_processed, y, preprocessor
        
    except Exception as e:
        logging.error(f"Error in data preprocessing: {e}")
        raise

def split_data(X, y, test_size=0.2, val_size=0.2, random_state=42):
    """Split data into train, validation, and test sets"""
    from sklearn.model_selection import train_test_split
    
    # First split: separate out test set
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)
    
    # Second split: separate train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size/(1-test_size), random_state=random_state)
    
    logging.info("Data splitting completed successfully")
    return X_train, X_val, X_test, y_train, y_val, y_test

def save_processed_data(X, y, file_path):
    """Save processed data to file"""
    try:
        # Combine features and target
        processed_data = pd.DataFrame(X)
        processed_data['target'] = y
        
        # Save to CSV
        processed_data.to_csv(file_path, index=False)
        logging.info(f"Processed data saved to {file_path}")
    except Exception as e:
        logging.error(f"Error saving processed data: {e}")
        raise
    