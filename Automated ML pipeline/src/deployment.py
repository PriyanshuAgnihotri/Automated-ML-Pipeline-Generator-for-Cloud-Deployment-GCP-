from google.cloud import aiplatform
import logging
import joblib

def deploy_to_ai_platform(model_path, config):
    """Deploy model to AI Platform"""
    try:
        # Initialize AI Platform
        aiplatform.init(project=config['deployment']['project_id'], 
                       location=config['deployment']['region'])
        
        # Upload model to AI Platform
        model = aiplatform.Model.upload(
            display_name=config['pipeline']['name'],
            artifact_uri=model_path,
            serving_container_image_uri=config['deployment']['serving_container_image']
        )
        
        # Deploy model to endpoint
        endpoint = model.deploy(
            machine_type=config['deployment']['machine_type'],
            min_replica_count=config['deployment']['min_replica_count'],
            max_replica_count=config['deployment']['max_replica_count']
        )
        
        logging.info(f"Model deployed successfully to {endpoint.resource_name}")
        return endpoint
    except Exception as e:
        logging.error(f"Error deploying model to AI Platform: {e}")
        raise

def create_cloud_function_trigger(bucket_name, function_url):
    """Create Cloud Storage trigger for Cloud Function"""
    from google.cloud import storage
    
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        
        # This would typically be set up in the Cloud Function configuration
        # For this example, we'll just log the information
        logging.info(f"Cloud Function trigger set up for bucket {bucket_name}")
        logging.info(f"Cloud Function URL: {function_url}")
        
        return True
    except Exception as e:
        logging.error(f"Error creating Cloud Function trigger: {e}")
        raise
    