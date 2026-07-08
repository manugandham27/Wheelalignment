import os
import pytest
import pandas as pd
from utils.helper import load_config
from data.data_loader import check_and_generate_mock_data, load_tyre_condition_images, load_rul_df

def test_config_loading():
    """Verify that config loads correctly and contains expected keys."""
    config = load_config()
    assert "data" in config
    assert "models" in config
    assert "outputs" in config
    assert config["models"]["wear_classifier"]["num_classes"] == 3

def test_mock_generation_and_loading():
    """Verify that data generation works and dataset directories exist."""
    config = load_config()
    
    # Run check and generate
    check_and_generate_mock_data(config)
    
    # Load images
    image_paths, labels = load_tyre_condition_images(config)
    assert len(image_paths) > 0
    assert len(labels) == len(image_paths)
    assert set(labels) == {0, 1, 2}
    
    # Load RUL df
    df = load_rul_df(config)
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] > 0
    assert "remaining_useful_life(km)" in df.columns
    assert "current_tread_depth(mm)" in df.columns
