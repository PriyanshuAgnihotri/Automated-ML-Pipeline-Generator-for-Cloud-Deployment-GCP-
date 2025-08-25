#!/usr/bin/env python3
"""
Script to run the complete ML pipeline
"""

import argparse
import logging
from src.pipeline import run_pipeline

def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run ML Pipeline')
    parser.add_argument('--pipeline-config', type=str, default='config/pipeline_config.yaml',
                       help='Path to pipeline configuration file')
    parser.add_argument('--model-config', type=str, default='config/model_config.yaml',
                       help='Path to model configuration file')
    
    args = parser.parse_args()
    
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