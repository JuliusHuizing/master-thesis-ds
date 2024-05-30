import argparse
import yaml
import logging
import subprocess
import sys
import time
import os
from sisa3d.yaml.yaml_utils import load_yaml_file
from sisa3d.clip import compute_clip
from sisa3d.results import save_results_to_csv

# Import date class from datetime module
import datetime 

def parse_arguments():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Run the pipeline with the specified configuration file.")
    parser.add_argument('--config', type=str, required=True, help="Path to the configuration YAML file.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    # Configure logging to write to stdout
    logging.basicConfig(handlers=[logging.StreamHandler(sys.stdout)],
                        level=logging.INFO,
                        format='[%(levelname)s] %(message)s')
    logging.getLogger().setLevel(logging.INFO)

    try:
        start_time = time.time()  # Start time for duration calculation

        logging.info("Loading configuration...")
        config = load_yaml_file(args.config)
        logging.info("✅ Configuration loaded.")
        
        # load paths
        logging.info("Loading paths...")
        DREAMGAUSSIAN_PATH = config["paths"]["dream_gaussian_repo_path"]
        INPUT_IMAGE_PATH = config["paths"]["input_image_path"]
        PREPROCCSING_OUTPUT_PATH = config["paths"]["preprocessing_output_path"]
        PREPROCESSED_IMAGE_PATH = os.path.join(PREPROCCSING_OUTPUT_PATH, os.path.basename(INPUT_IMAGE_PATH).split(".")[0] + "_rgba.png")
        MODEL_OUTPUT_PATH = config["paths"]["model_output_path"]
        STAGE_1_IMAGES_PATH = config["paths"]["stage_1_images_output_path"]
        STAGE_1_CLIP_SCORES_OUTPUT_PATH = config["paths"]["stage_1_clip_scores_output_path"]
        logging.info("✅ Paths loaded.")
        
        logging.info("Creating paths if they don't exist...")
        for path in [PREPROCCSING_OUTPUT_PATH, MODEL_OUTPUT_PATH, STAGE_1_IMAGES_PATH]:
            os.makedirs(path, exist_ok=True)
        logging.info("✅ Paths created.")
        
        logging.info("Initializing preprocessing pipeline...")
        preprocessing_config = config["preprocess"]
        
        logging.info("Running preprocessing pipeline...")
        
        # Get values from your preprocessing_config dictionary
        size = preprocessing_config['size']
        recenter = preprocessing_config['recenter']
        border_ratio = preprocessing_config['border_ratio']
        # Construct the command with arguments
        command = [
            'python', os.path.join(DREAMGAUSSIAN_PATH, "process.py"),
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
            "python", os.path.join(DREAMGAUSSIAN_PATH, "main.py"), 
            "--config", args.config, 
            f"input={PREPROCESSED_IMAGE_PATH}", 
            "save_path=name"
        ]
        result = subprocess.run(command, check=True, text=True, capture_output=True)
        logging.info("✅ DreamGaussian pipeline complete.")
        
        logging.info("Running Evaluation pipeline...")
        
        clip_score = compute_clip(f"{STAGE_1_IMAGES_PATH}generated", f"{STAGE_1_IMAGES_PATH}reference").item()
        clip_scores = compute_clip(f"{STAGE_1_IMAGES_PATH}generated", f"{STAGE_1_IMAGES_PATH}reference", average=False)
        clip_scores = [x.item() for x in clip_scores]
        
        

        
        # Calculate duration
        duration = time.time() - start_time

        # Save results to CSV
        csv_path = STAGE_1_CLIP_SCORES_OUTPUT_PATH  # Path from the config
            
        # For privacy, remove the paths from the config
        hyperparameters = {k: v for k, v in config.items() if k != 'paths'}
        
        # Convert hyperparameters to a string for CSV
        hyperparameters_str = yaml.dump(hyperparameters)
        # for reproducibility, save the full config to the CSV
        full_config = yaml.dump(config)
        elongation = config["dreamgaussian"]["regularize"]["elongation"]
        compactness = config["dreamgaussian"]["regularize"]["compactness"]
        opacity = config["dreamgaussian"]["regularize"]["opacity"]
        
        row = {
            'clip_score': clip_score,
            'elongation': elongation,
            'compactness': compactness,
            'opacity': opacity,
            'clip_scores': clip_scores,
            'duration': duration,
            'input_image_path': INPUT_IMAGE_PATH,
            'hyperparameters': hyperparameters_str,
            "full_config": full_config,
        }
        
        save_results_to_csv(csv_path, row)
        logging.info(f"... Results saved to {csv_path}.")
        logging.info("✅ Evaluation pipeline complete.")
        
        logging.info("Running DreamGaussian Stage 2 pipeline...")
        command = [
            "python", os.path.join(DREAMGAUSSIAN_PATH, "main2.py"), 
            "--config", args.config, 
            f"input={PREPROCESSED_IMAGE_PATH}", 
            "save_path=name"
        ]
        result = subprocess.run(command, check=True, text=True, capture_output=True)
        logging.info("✅ DreamGaussian pipeline complete.")
    
    except subprocess.CalledProcessError as cpe:
        # Log the output and error output from the subprocess
        logging.error("❌ Subprocess error with non-zero exit status %s", cpe.returncode)
        if cpe.stdout:
            logging.error("Standard output of the subprocess: %s", cpe.stdout)
        if cpe.stderr:
            logging.error("Standard error of the subprocess: %s", cpe.stderr)
        raise cpe
    except Exception as e:
        logging.error(f"❌ Error in pipeline: {e}")
        raise e
