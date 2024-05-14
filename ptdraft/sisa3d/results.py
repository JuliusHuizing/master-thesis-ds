
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
def save_results_to_csv(csv_path, clip_score, duration, config):
    header = ['clip score', 'duration', 'hyperparameters']
    
    # For privacy, remove the paths from the config
    hyperparameters = {k: v for k, v in config.items() if k != 'paths'}
    
    # Convert hyperparameters to a string for CSV
    hyperparameters_str = yaml.dump(hyperparameters)
    
    # Check if the CSV file exists
    file_exists = os.path.isfile(csv_path)
    
    with open(csv_path, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=header)
        
        # Write the header only if the file doesn't exist
        if not file_exists:
            writer.writeheader()
        
        # Write the row with results
        writer.writerow({
            'clip score': clip_score,
            'duration': duration,
            'hyperparameters': hyperparameters_str
        })
