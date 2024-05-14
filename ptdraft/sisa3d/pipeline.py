import concurrent.futures
import subprocess
import os
import logging

def run_pipeline_for_config(config_path):
    """
    Run the pipeline.py script with a given configuration file.

    Args:
        config_path (str): Path to the configuration file.
    """
    try:
        # Set the environment variable for the configuration file
        env = os.environ.copy()
        env['CONFIG_PATH'] = config_path
        
        # Run the pipeline script
        result = subprocess.run(['python', 'pipeline.py'], env=env, capture_output=True, text=True)
        
        # Log the result
        logging.info(f"Completed {config_path} with return code {result.returncode}")
        
        if result.stdout:
            logging.info(f"Standard output: {result.stdout}")
        if result.stderr:
            logging.error(f"Standard error: {result.stderr}")
            
    except subprocess.CalledProcessError as e:
        logging.error(f"Pipeline for {config_path} failed: {e}")
    except Exception as e:
        logging.error(f"Error running pipeline for {config_path}: {e}")
