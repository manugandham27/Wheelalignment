import os
import pytest
from utils.helper import load_config
from pipeline.predict_pipeline import UnifiedPredictionPipeline

def test_pipeline_integration():
    """Verify that UnifiedPredictionPipeline can run full prediction and output correct keys."""
    config = load_config()
    raw_dir = config["data"]["raw_dir"]
    
    mock_image_path = os.path.join(raw_dir, config["data"]["tyre_condition"], "Serviceable", "mock_serviceable_0.jpg")
    
    # Ensure mock data is present
    from data.data_loader import check_and_generate_mock_data
    check_and_generate_mock_data(config)
    
    if os.path.exists(mock_image_path):
        pipeline = UnifiedPredictionPipeline()
        
        # Test prediction without sensor readings (uses fallback fusion mapping)
        res = pipeline.predict(mock_image_path)
        
        expected_keys = [
            "wear_class", "estimated_tread_depth_mm", "predicted_rul_km",
            "alignment_flag", "alignment_confidence", "explanation_heatmap_path",
            "diagnosis"
        ]
        
        for key in expected_keys:
            assert key in res
            
        assert res["wear_class"] in ["New", "Serviceable", "Unusable"]
        assert isinstance(res["estimated_tread_depth_mm"], float)
        assert isinstance(res["predicted_rul_km"], float)
        assert isinstance(res["alignment_flag"], bool)
    else:
        pytest.skip("Mock image not found for pipeline integration test.")
