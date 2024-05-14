
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
def save_results_to_csv(csv_path, row: dict):
    header = row.keys()

    
    # Check if the CSV file exists
    file_exists = os.path.isfile(csv_path)
    
    with open(csv_path, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=header)
        
        # Write the header only if the file doesn't exist
        if not file_exists:
            writer.writeheader()
        
        # Write the row with results
        writer.writerow(row)
