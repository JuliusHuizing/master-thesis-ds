import yaml
import itertools
import os

def flatten_dict(d, parent_key='', sep='.'):
    """
    Flatten a nested dictionary.

    Args:
        d (dict): The dictionary to flatten.
        parent_key (str, optional): The base key string for the dictionary. Defaults to ''.
        sep (str, optional): Separator for the keys. Defaults to '.'.

    Returns:
        dict: Flattened dictionary.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def unflatten_dict(d, sep='.'):
    """
    Unflatten a dictionary.

    Args:
        d (dict): The dictionary to unflatten.
        sep (str, optional): Separator for the keys. Defaults to '.'.

    Returns:
        dict: Unflattened dictionary.
    """
    result_dict = {}
    for k, v in d.items():
        keys = k.split(sep)
        d = result_dict
        for key in keys[:-1]:
            if key not in d:
                d[key] = {}
            d = d[key]
        d[keys[-1]] = v
    return result_dict

def deep_update_dict(source, overrides):
    """
    Recursively update a nested dictionary.

    Args:
        source (dict): The source dictionary to update.
        overrides (dict): The dictionary with values to update.

    Returns:
        dict: The updated dictionary.
    """
    for key, value in overrides.items():
        if isinstance(value, dict) and key in source:
            source[key] = deep_update_dict(source.get(key, {}), value)
        else:
            source[key] = value
    return source

def create_grid_search_config_files(path_to_default_config_yaml, path_to_grid_search_yaml, output_dir):
    """
    Create configuration files for each combination of hyperparameters specified in the grid search configuration.

    Args:
        path_to_default_config_yaml (str): Path to the default configuration YAML file.
        path_to_grid_search_yaml (str): Path to the grid search configuration YAML file.
        output_dir: Directory to save the generated configuration files.
    """
    # Load the default configuration file
    with open(path_to_default_config_yaml, 'r') as file:
        default_config = yaml.safe_load(file)

    # Load the grid search configuration file
    with open(path_to_grid_search_yaml, 'r') as file:
        grid_search_config = yaml.safe_load(file)
        
        
    image_dir = grid_search_config['image_dir']
    # List to store the paths of all .png files
    image_paths = []
    # Walk through the directory
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.endswith('.png'):
                # Construct the full path and add it to the list
                image_paths.append(os.path.join(root, file))

  
    # Extract hyperparameters and their values from the grid search config
    hyperparameters = grid_search_config['hyperparameters']
    
    # Flatten the hyperparameters dictionary for easy combination
    flattened_hyperparameters = flatten_dict(hyperparameters)
    
    # Create all combinations of hyperparameters
    keys, values = zip(*flattened_hyperparameters.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    # Create output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Generate a config file for each combination
    for image_idx, image_path in enumerate(image_paths):
        for idx, combo in enumerate(combinations):
            # Create a new config based on the default config
            new_config = yaml.safe_load(yaml.dump(default_config))  # Deep copy
            
            # Unflatten the combination to get nested structure
            nested_combo = unflatten_dict(combo)
            
            # Update the config with the current combination of hyperparameters
            new_config = deep_update_dict(new_config, nested_combo)
            
            new_config['paths']['input_image_path'] = image_path
            
            # Define the output path
            output_path = os.path.join(output_dir, f'temporary_{idx}_for_img_{image_idx}.config.yaml')
            
            with open(output_path, 'w') as file:
                yaml.dump(new_config, file)
    
    print(f'Generated {len(combinations)} configuration files in {output_dir} for {len(image_paths)} images.')

