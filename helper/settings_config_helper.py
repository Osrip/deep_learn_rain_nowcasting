import os
import yaml

def load_yaml_config(filename):
    with open(filename, 'r') as f:
        return yaml.safe_load(f)

def load_settings(mode):
    # Load settings based on mode.
    # In cluster mode, use the cluster_default config.
    if mode == 'cluster':
        settings = load_yaml_config('configs/cluster_default.yml')
    # In local mode, load the cluster_default then update with the local overrides if available.
    elif mode == 'local':
        settings = load_yaml_config('configs/cluster_default.yml')
        local_overrides = load_yaml_config('configs/local_config.yml')
        settings.update(local_overrides)
    # In debug mode, use the debug_config file.
    elif mode == 'debug':
        settings = load_yaml_config('configs/cluster_default.yml')
        debug_overrides = load_yaml_config('configs/debug_config.yml')
        settings.update(debug_overrides)
    else:
        raise ValueError("Invalid mode. Choose from 'cluster', 'local', or 'debug'.")
    return settings