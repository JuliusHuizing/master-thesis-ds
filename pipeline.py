
import yaml
import runpy
from utils.yaml_utils import load_yaml_file
import logging
import runpy
import subprocess

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.getLogger().setLevel(logging.INFO)
    # Configure logging
    try:
        logging.info("Loading configuration...")
        config = load_yaml_file('config.yaml')
        logging.info("✅ Configuration loaded.")
        
        # load paths
        logging.info("Loading paths...")
        DREAMGAUSSIAN_PATH = config["paths"]["dream_gaussian_repo_path"]
        logging.info("✅ Paths loaded.")
        
        
        
        logging.info("Initializing preprocessing pipeline...")
        # isolate integration step as separate phaase
        preprocessing_config = config["preprocess"]
        
        logging.info("Running preprocessing pipeline...")
        
        # Get values from your preprocessing_config dictionary
        size = preprocessing_config['size']
        recenter = preprocessing_config['recenter']
        # Construct the command with arguments
        command = [
            'python', DREAMGAUSSIAN_PATH+"process.py",
            '--size', str(size),
            '--border_ratio',
            '--recenter', str(recenter)
        ]

        # Execute the command
        result = subprocess.run(command)
        
        # runpy.run_path(DREAMGAUSSIAN_PATH + f"/process.py --size {preprocessing_config['size']} --border_ratio --recenter {preprocessing_config['recenter']}")
        logging.info("✅ Preprocessing pipeline complete.")
        
        
    except Exception as e:
        logging.error(f" ❌ Error in pipeline: {e}")
        raise e
    
    
    
    
    
    
    
    

        
        
        
        
            
            
            
        
    
    
    