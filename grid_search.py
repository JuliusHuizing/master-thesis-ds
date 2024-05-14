import argparse
import os
import subprocess
import logging
import sys
from tqdm import tqdm
from sisa3d.gridsearch import create_grid_search_config_files

def parse_arguments():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Run grid search over multiple configuration files created by a grid search config file")
    parser.add_argument('--default_config', type=str, required=True, help="default config file")
    parser.add_argument('--gridsearch_config', type=str, required=True, help="grid search config file")
    return parser.parse_args()

def run_pipeline(config_path):
    """
    Run the pipeline with the specified configuration file.

    Args:
        config_path (str): Path to the configuration YAML file.
    """
    try:
        logging.info(f"Running pipeline with config: {config_path}")
        subprocess.run(["python", "pipeline.py", "--config", config_path], check=True)
        logging.info(f"✅ Pipeline run complete for config: {config_path}")
    except subprocess.CalledProcessError as cpe:
        logging.error(f"❌ Subprocess error with non-zero exit status {cpe.returncode} for config: {config_path}")
        if cpe.stdout:
            logging.error(f"Standard output of the subprocess: {cpe.stdout.decode()}")
        if cpe.stderr:
            logging.error(f"Standard error of the subprocess: {cpe.stderr.decode()}")
        raise cpe
    except Exception as e:
        logging.error(f"❌ Error in running pipeline for config: {config_path} - {e}")
        raise e

if __name__ == "__main__":
    args = parse_arguments()

    # Configure logging to write to stdout
    logging.basicConfig(handlers=[logging.StreamHandler(sys.stdout)],
                        level=logging.INFO,
                        format='[%(levelname)s] %(message)s')
    logging.getLogger().setLevel(logging.INFO)

    # List all YAML files in the provided directory
    tmp_configs_dir = "tmp_configs"
    logging.info(f"Creating temporary directory for config files: {tmp_configs_dir}")
    create_grid_search_config_files(args.default_config, args.gridsearch_config, tmp_configs_dir)
    logging.info(f"✅ Temporary directory created with config files: {tmp_configs_dir}")
    config_files = [f for f in os.listdir(tmp_configs_dir) if f.endswith('.yaml')]
    if not config_files:
        logging.error(f"No YAML configuration files found in directory: {args.configs_dir}")
        sys.exit(1)


    logging.info(f"Running sequantial grid search with {len(config_files)} configuration files...")
    # Run pipeline.py for each configuration file sequentially with a progress bar
    for config_file in tqdm(config_files, desc="Running grid search"):
        config_path = os.path.join(args.configs_dir, config_file)
        run_pipeline(config_path)
    logging.info("✅ Grid search complete.")
    logging.info("Cleaning up...")
    # delete tmp dicts
    subprocess.run(["rm", "-rf", tmp_configs_dir])
    logging.info("✅ Done")
