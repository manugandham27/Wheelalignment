import os
import yaml
import torch

def load_config(config_path="config.yaml"):
    """
    Load project configuration from a YAML file.
    """
    if not os.path.exists(config_path):
        # Default fallback config if running from a different directory
        parent_config = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")
        if os.path.exists(parent_config):
            config_path = parent_config
        else:
            raise FileNotFoundError(f"Config file not found at {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def get_device(config=None):
    """
    Detect the best available PyTorch device.
    Supports CUDA, MPS (Apple Silicon), and CPU.
    Can be overridden in config.yaml under general.device.
    """
    if config and "general" in config and config["general"].get("device"):
        cfg_device = config["general"]["device"]
        if cfg_device in ["cuda", "mps", "cpu"]:
            if cfg_device == "cuda" and torch.cuda.is_available():
                return torch.device("cuda")
            elif cfg_device == "mps" and torch.backends.mps.is_available():
                return torch.device("mps")
            elif cfg_device == "cpu":
                return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def create_dirs_if_not_exist(config):
    """
    Create necessary output directories specified in config.
    """
    output_dir = config.get("outputs", {}).get("output_dir", "outputs")
    heatmap_dir = config.get("outputs", {}).get("heatmap_dir", "outputs/heatmaps")
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(heatmap_dir, exist_ok=True)

    # Also make sure raw directory paths exist for the user to download data into
    raw_dir = config.get("data", {}).get("raw_dir", "data/raw")
    datasets = ["tyrenet", "tyre_quality", "tyre_condition", "tire_texture", "synthetic_rul"]
    for dataset in datasets:
        dataset_subdir = config.get("data", {}).get(dataset, dataset)
        os.makedirs(os.path.join(raw_dir, dataset_subdir), exist_ok=True)
