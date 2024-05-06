import yaml
import importlib
import os
import yaml

def load_yaml_file(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)
