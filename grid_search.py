import yaml
import itertools
import subprocess
import copy

# Load the default configuration
with open('config.yaml', 'r') as file:
    default_config = yaml.safe_load(file)

# Define the grid of parameters to test
param_grid = {
    'training_batch_size': [1, 2, 4],
    'training_lambda_zero123': [0, 0.5, 1],
    'input_density_thresh': [0.5, 1, 1.5]
}

# Function to update nested dictionaries
def update_config(base, changes):
    for key, value in changes.items():
        if isinstance(value, dict) and key in base:
            update_config(base[key], value)
        else:
            base[key] = value

# Iterate over all combinations of grid parameters
for values in itertools.product(*param_grid.values()):
    # Create a temporary config for this combination
    temp_config = copy.deepcopy(default_config)
    config_updates = dict(zip(param_grid.keys(), values))
    
    # Convert flat keys to nested dictionary updates
    nested_updates = {}
    for key, value in config_updates.items():
        keys = key.split('_')
        current = nested_updates
        for part in keys[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[keys[-1]] = value

    # Update the configuration
    update_config(temp_config, nested_updates)
    
    # Save the modified configuration to a temporary YAML file
    temp_yaml_path = 'temp_config.yaml'
    with open(temp_yaml_path, 'w') as file:
        yaml.dump(temp_config, file)
    
    # Run the generation process using the modified config
    # subprocess.run(['python', 'your_generation_script.py', '--config', temp_yaml_path])

    # Optionally, handle outputs and logging here
