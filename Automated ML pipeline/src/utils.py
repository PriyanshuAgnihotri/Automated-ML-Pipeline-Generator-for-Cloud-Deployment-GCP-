import yaml
import logging
from google.cloud import storage

def load_config(config_path):
    """Load YAML configuration file"""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        logging.error(f"Error loading config file: {e}")
        raise

def download_from_gcs(bucket_name, source_blob_name, destination_file_name):
    """Download a file from Google Cloud Storage"""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    logging.info(f"Downloaded {source_blob_name} to {destination_file_name}.")

def upload_to_gcs(bucket_name, source_file_name, destination_blob_name):
    """Upload a file to Google Cloud Storage"""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    logging.info(f"Uploaded {source_file_name} to {destination_blob_name}.")

def validate_config(config, config_type):
    """Validate configuration structure"""
    if config_type == "pipeline":
        required_sections = ["data_processing", "training", "validation", "deployment"]
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required section in pipeline config: {section}")
    
    return True