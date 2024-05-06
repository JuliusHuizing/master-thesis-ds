
import yaml
import runpy
from utils.yaml_utils import load_yaml_file
import logging
import runpy

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
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
        preprocessing_config = config["preprocess"]["integrate"]
        
        logging.info("Running preprocessing pipeline...")
        runpy.run_path(DREAMGAUSSIAN_PATH + f"/process.py --size {preprocessing_config['size']} --border_ratio --recenter {preprocessing_config['recenter']}")
        logging.info("✅ Preprocessing pipeline complete.")
        
        
    except Exception as e:
        logging.error(f" ❌ Error in pipeline: {e}")
        raise e
    
    
    
    
    
    
    
    

        
        
        
        
            
            
            
        
    
    
    