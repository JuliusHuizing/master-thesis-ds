
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
import matplotlib.pyplot as plt
import pandas as pd


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






def plot_scatter_from_csv(file_path, output_path):
    # Load the CSV file
    df = pd.read_csv(file_path)
    
    # Create parent directory if it does not exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Scatter plots for clip_score vs elongation, compactness, and opacity
    plt.figure(figsize=(18, 5))

    # Scatter Plot: Clip Score vs Elongation
    plt.subplot(1, 3, 1)
    plt.scatter(df['elongation'], df['clip_score'], edgecolors='w', alpha=0.7)
    plt.title('Scatter Plot: Clip Score vs Elongation')
    plt.xlabel('Elongation')
    plt.ylabel('Clip Score')

    # Scatter Plot: Clip Score vs Compactness
    plt.subplot(1, 3, 2)
    plt.scatter(df['compactness'], df['clip_score'], edgecolors='w', alpha=0.7)
    plt.title('Scatter Plot: Clip Score vs Compactness')
    plt.xlabel('Compactness')
    plt.ylabel('Clip Score')

    # Scatter Plot: Clip Score vs Opacity
    plt.subplot(1, 3, 3)
    plt.scatter(df['opacity'], df['clip_score'], edgecolors='w', alpha=0.7)
    plt.title('Scatter Plot: Clip Score vs Opacity')
    plt.xlabel('Opacity')
    plt.ylabel('Clip Score')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()



