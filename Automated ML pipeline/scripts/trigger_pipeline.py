#!/usr/bin/env python3
"""
Script to trigger pipeline execution based on Cloud Storage events
"""

import argparse
import logging
from src.pipeline import run_pipeline

def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Trigger ML Pipeline based on GCS event')
    parser.add_argument('--bucket', type=str, required=True,
                       help='GCS bucket name')
    parser.add_argument('--file', type=str, required=True,
                       help='File path that triggered the event')
    parser.add_argument('--pipeline-config', type=str, default='config/pipeline_config.yaml',
                       help='Path to pipeline configuration file')
    parser.add_argument('--model-config', type=str, default='config/model_config.yaml',
                       help='Path to model configuration file')
    
    args = parser.parse_args()
    
    logging.info(f"Pipeline triggered by file: gs://{args.bucket}/{args.file}")
    
    # Run pipeline
    try:
        success, metrics = run_pipeline(args.pipeline_config, args.model_config)
        if success:
            logging.info("Pipeline completed successfully")
            logging.info(f"Final metrics: {metrics}")
        else:
            logging.error("Pipeline failed validation")
            exit(1)
    except Exception as e:
        logging.error(f"Pipeline execution failed: {e}")
        exit(1)

if __name__ == "__main__":
    main()
    