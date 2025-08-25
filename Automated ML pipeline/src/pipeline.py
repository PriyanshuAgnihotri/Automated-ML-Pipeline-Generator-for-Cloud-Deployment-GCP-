import logging
from .data_processing import load_data, preprocess_data, split_data, save_processed_data
from .model_training import train_model, evaluate_model, save_model
from .model_validation import validate_model
from .deployment import deploy_to_ai_platform
from .utils import load_config, validate_config, download_from_gcs, upload_to_gcs

def run_pipeline(config_path, model_config_path):
    """Run the complete ML pipeline"""
    try:
        # Load configurations
        pipeline_config = load_config(config_path)
        model_config = load_config(model_config_path)
        
        # Validate configurations
        validate_config(pipeline_config, "pipeline")
        validate_config(model_config, "model")
        
        # Data processing stage
        logging.info("Starting data processing stage")
        data = load_data(pipeline_config['data_processing']['input_path'])
        X_processed, y, preprocessor = preprocess_data(data, pipeline_config)
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(
            X_processed, y, 
            test_size=0.2, 
            val_size=0.2
        )
        
        # Save processed data
        save_processed_data(X_train, y_train, pipeline_config['data_processing']['output_path'] + 'train.csv')
        save_processed_data(X_val, y_val, pipeline_config['data_processing']['output_path'] + 'val.csv')
        save_processed_data(X_test, y_test, pipeline_config['data_processing']['output_path'] + 'test.csv')
        
        # Training stage
        logging.info("Starting model training stage")
        model = train_model(
            X_train, y_train, 
            pipeline_config['training']['model_type'],
            pipeline_config['training']['hyperparameters'],
            model_config
        )
        
        # Evaluate model
        accuracy, predictions = evaluate_model(model, X_val, y_val)
        
        # Save model
        model_path = pipeline_config['training']['model_output_path'] + 'model.joblib'
        save_model(model, model_path)
        
        # Validation stage
        logging.info("Starting model validation stage")
        validation_passed, metrics = validate_model(
            model, X_test, y_test,
            pipeline_config['validation']['metrics'],
            pipeline_config['validation']['thresholds']
        )
        
        if not validation_passed:
            logging.error("Model validation failed. Deployment aborted.")
            return False, metrics
        
        # Deployment stage
        logging.info("Starting deployment stage")
        if pipeline_config['deployment']['platform'] == 'ai_platform':
            endpoint = deploy_to_ai_platform(model_path, pipeline_config)
            logging.info(f"Model deployed successfully to AI Platform: {endpoint.resource_name}")
        else:
            logging.warning("Only AI Platform deployment is currently supported")
        
        logging.info("Pipeline execution completed successfully")
        return True, metrics
        
    except Exception as e:
        logging.error(f"Pipeline execution failed: {e}")
        raise
    