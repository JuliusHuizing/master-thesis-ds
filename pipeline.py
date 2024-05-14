
import yaml
import runpy
from sisa3d.yaml.yaml_utils import load_yaml_file
import logging
import runpy
import subprocess
import sys
import time
import csv
import os
from sisa3d.clip import compute_clip
from sisa3d.results import save_results_to_csv


if __name__ == "__main__":
    # Configure logging to write to stdout
    logging.basicConfig(handlers=[logging.StreamHandler(sys.stdout)],
                    level=logging.INFO,
                    format='[%(levelname)s] %(message)s')
    # logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.getLogger().setLevel(logging.INFO)
    # Configure logging
    try:
        start_time = time.time()  # Start time for duration calculation

        logging.info("Loading configuration...")
        config = load_yaml_file('config.yaml')
        logging.info("✅ Configuration loaded.")
        
        # load paths
        logging.info("Loading paths...")
        DREAMGAUSSIAN_PATH = config["paths"]["dream_gaussian_repo_path"]
        INPUT_IMAGE_PATH = config["paths"]["input_image_path"]
        PREPROCCSING_OUTPUT_PATH = config["paths"]["preprocessing_output_path"]
        PREPROCESSED_IMAGE_PATH = PREPROCCSING_OUTPUT_PATH + INPUT_IMAGE_PATH.split("/")[-1].split(".")[0] + "_rgba.png"
        MODEL_OUTPUT_PATH = config["paths"]["model_output_path"]
        STAGE_1_IMAGES_PATH = config["paths"]["stage_1_images_output_path"]
        STAGE_1_CLIP_SCORES_OUTPUT_PATH = config["paths"]["stage_1_clip_scores_output_path"]
        logging.info("✅ Paths loaded.")
        
        logging.info("Creating paths if they don't exist...")
        for path in [PREPROCCSING_OUTPUT_PATH, MODEL_OUTPUT_PATH, STAGE_1_IMAGES_PATH]:
            subprocess.run(["mkdir", "-p", path])
        logging.info("✅ Paths created.")
        
        logging.info("Initializing preprocessing pipeline...")
        # isolate integration step as separate phaase
        preprocessing_config = config["preprocess"]
        
        logging.info("Running preprocessing pipeline...")
        
        # Get values from your preprocessing_config dictionary
        size = preprocessing_config['size']
        recenter = preprocessing_config['recenter']
        border_ratio = preprocessing_config['border_ratio']
        # Construct the command with arguments
        command = [
            'python', DREAMGAUSSIAN_PATH+"process.py",
            INPUT_IMAGE_PATH,
            '--output_dir', PREPROCCSING_OUTPUT_PATH,
            '--size', str(size),
            '--border_ratio', str(border_ratio),
            '--recenter', str(recenter)
        ]

        # Execute the command
        result = subprocess.run(command, check=True, text=True, capture_output=True)
        logging.info("✅ Preprocessing pipeline complete.")
        
        logging.info("Running DreamGaussian Stage 1 pipeline...")
        command = [
            "python", DREAMGAUSSIAN_PATH+"main.py", 
            f"--config", "config.yaml", 
            f"input={PREPROCESSED_IMAGE_PATH}", 
            "save_path=name"
        ]
        result = subprocess.run(command, check=True, text=True, capture_output=True)
        logging.info("✅ DreamGaussian pipeline complete.")
        
        

        logging.info("Running Evaluation pipeline...")
        clip_score = compute_clip(f"{STAGE_1_IMAGES_PATH}generated", 
                     f"{STAGE_1_IMAGES_PATH}reference")
        
        
        result = subprocess.run(command, check=True, text=True, capture_output=True)       
        
        # Calculate duration
        duration = time.time() - start_time

        # Save results to CSV
        csv_path = 'results/stage_1/clip_scores.csv'  # You can change this to your desired path
        save_results_to_csv(csv_path, clip_score, duration, config)
        logging.info(f" ... Results saved to {csv_path}.")
        logging.info("✅ Evaluation pipeline complete.")


        # logging.info("Running DreamGaussian Stage 2 pipeline...")
        # # python main2.py --config configs/image.yaml input=data/name_rgba.png save_path=name
        # command = [
        #     "python", DREAMGAUSSIAN_PATH+"main2.py",
        #     f"--config", f"{DREAMGAUSSIAN_PATH}configs/image.yaml",
        #     f"input={PREPROCESSED_IMAGE_PATH}",
        # ]

    
        
    except subprocess.CalledProcessError as cpe:
        # Log the output and error output from the subprocess
        logging.error("❌ Subprocess error with non-zero exit status %s", cpe.returncode)
        if cpe.stdout:
            logging.error("Standard output of the subprocess: %s", cpe.stdout)
        if cpe.stderr:
            logging.error("Standard error of the subprocess: %s", cpe.stderr)
        raise cpe
    except Exception as e:
        logging.error(f" ❌ Error in pipeline: {e}")
        raise e
        
