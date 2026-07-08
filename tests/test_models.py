import os
import torch
import numpy as np
import pandas as pd
import pytest

from utils.helper import load_config
from models.wear_classifier.model import DualHeadTireCNN
from models.rul_predictor.model import TireRULPredictor
from models.alignment_heuristic.analyzer import analyze_wear_asymmetry

def test_cnn_forward_shape():
    """Verify that DualHeadTireCNN returns outputs of correct shapes."""
    model = DualHeadTireCNN(backbone_name="resnet18", pretrained=False, num_classes=3)
    
    # Batch size 2, 3 channels, 224x224
    dummy_input = torch.randn(2, 3, 224, 224)
    cls_out, reg_out = model(dummy_input)
    
    assert cls_out.shape == (2, 3)
    assert reg_out.shape == (2,)

def test_rul_predictor_fit_predict():
    """Test standard fit and predict functionality of TireRULPredictor on dummy DataFrame."""
    predictor = TireRULPredictor(n_estimators=5, max_depth=3)
    
    # Create simple dummy dataframe with exact required columns
    dummy_features = {
        "vehicle_model": ["Sedan", "SUV"],
        "fuel_type": ["Petrol", "Diesel"],
        "transmission_type": ["Automatic", "Manual"],
        "country": ["Germany", "USA"],
        "maximum_power(hp)": [150, 200],
        "maximum_torque(N/m)": [220, 300],
        "maximum_speed(km/h)": [200, 220],
        "vehicle_acceleration(0-100 km/h in seconds)": [8.5, 7.2],
        "vehicle_mileage(mpg)": [30.0, 25.0],
        "vehicle_sprung_mass(kg)": [1500, 1800],
        "steering_radius(m)": [5.5, 5.8],
        "axle_type(driven/dead)": ["driven", "dead"],
        "tyre_brand": ["Michelin", "Bridgestone"],
        "tyre_size": ["205/55R16", "225/65R17"],
        "tread_material": ["Silica Compound", "Carbon Black"],
        "tread_pattern": ["Symmetric", "Asymmetric"],
        "tyre_camber_angle(degree)": [0.0, -1.0],
        "standard_tread_depth(mm)": [8.0, 8.0],
        "retreaded": ["No", "No"],
        "road_condition": ["Smooth", "Rough"],
        "weather_condition": ["Dry", "Humid"],
        "expected_tyre_life(km)": [50000, 60000],
        "kilometers_driven(km)": [20000, 30000]
    }
    X = pd.DataFrame(dummy_features)
    
    # Targets
    y = pd.DataFrame({
        "remaining_useful_life(km)": [30000.0, 25000.0],
        "current_tread_depth(mm)": [5.2, 4.8]
    })
    
    predictor.train(X, y)
    preds = predictor.predict(X)
    
    assert preds.shape == (2, 2)
    assert isinstance(preds, np.ndarray)

def test_alignment_heuristic():
    """Verify that classical CV analyzer runs and successfully outputs asymmetry values."""
    config = load_config()
    raw_dir = config["data"]["raw_dir"]
    
    # Use a generated mock image
    mock_image_dir = os.path.join(raw_dir, config["data"]["tyre_condition"], "New")
    os.makedirs(mock_image_dir, exist_ok=True)
    mock_image_path = os.path.join(mock_image_dir, "mock_new_0.jpg")
    
    # Force mock generation if not already done
    from data.data_loader import check_and_generate_mock_data
    check_and_generate_mock_data(config)
    
    if os.path.exists(mock_image_path):
        results = analyze_wear_asymmetry(mock_image_path, config)
        assert "asymmetry_score" in results
        assert "alignment_flag" in results
        assert "diagnosis" in results
        assert isinstance(results["alignment_flag"], bool)
        assert 0.0 <= results["alignment_confidence"] <= 1.0
    else:
        pytest.skip("Mock image path not found for CV test.")
