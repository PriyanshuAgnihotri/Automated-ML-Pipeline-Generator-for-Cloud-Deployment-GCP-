import importlib
import logging
import joblib
from sklearn.metrics import accuracy_score

def load_model_class(model_type, model_config):
    """Dynamically load model class based on configuration"""
    try:
        model_info = model_config['models'][model_type]
        module = importlib.import_module(model_info['package'])
        model_class = getattr(module, model_info['class'])
        return model_class
    except Exception as e:
        logging.error(f"Error loading model class: {e}")
        raise

def train_model(X_train, y_train, model_type, hyperparameters, model_config):
    """Train model based on configuration"""
    try:
        # Load model class
        model_class = load_model_class(model_type, model_config)
        
        # Create model instance with hyperparameters
        model = model_class(**hyperparameters)
        
        # Train model
        model.fit(X_train, y_train)
        
        logging.info(f"Model training completed successfully for {model_type}")
        return model
    except Exception as e:
        logging.error(f"Error in model training: {e}")
        raise

def evaluate_model(model, X_val, y_val):
    """Evaluate model performance"""
    try:
        predictions = model.predict(X_val)
        accuracy = accuracy_score(y_val, predictions)
        
        logging.info(f"Model evaluation completed with accuracy: {accuracy}")
        return accuracy, predictions
    except Exception as e:
        logging.error(f"Error in model evaluation: {e}")
        raise

def save_model(model, model_path):
    """Save trained model to file"""
    try:
        joblib.dump(model, model_path)
        logging.info(f"Model saved to {model_path}")
    except Exception as e:
        logging.error(f"Error saving model: {e}")
        raise