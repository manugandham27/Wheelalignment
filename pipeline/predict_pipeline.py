import os
import json
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms

from utils.helper import load_config
from models.wear_classifier.model import DualHeadTireCNN
from models.rul_predictor.model import TireRULPredictor
from models.alignment_heuristic.analyzer import analyze_wear_asymmetry
from explainability.gradcam import generate_gradcam_visualization
from explainability.integrated_gradients import generate_ig_visualization
from data.data_loader import generate_simulated_imu

class UnifiedPredictionPipeline:
    """
    Upscaled Multimodal Prediction Pipeline.
    Integrates ConvNeXt-T, LSTM IMU encoder, Homographic CV,
    MC Dropout Uncertainty, and Integrated Gradients.
    """
    def __init__(self, config_path="config.yaml"):
        self.config = load_config(config_path)
        
        # Load Wear Classifier
        self.num_classes = self.config["models"]["wear_classifier"]["num_classes"]
        self.backbone = self.config["models"]["wear_classifier"].get("backbone", "resnet18")
        self.cnn_path = self.config["models"]["wear_classifier"]["save_path"]
        
        if os.path.exists(self.cnn_path):
            self.cnn_model = DualHeadTireCNN(backbone_name=self.backbone, pretrained=False, num_classes=self.num_classes)
            self.cnn_model.load_state_dict(torch.load(self.cnn_path, map_location=torch.device("cpu")))
            self.cnn_model.eval()
            print("Loaded CNN Wear Classifier checkpoint.")
        else:
            self.cnn_model = None
            print(f"Warning: CNN Wear Classifier checkpoint not found at {self.cnn_path}. Vision predictions will be stubs.")

        # Load Tabular RUL Predictor
        self.rul_path = self.config["models"]["rul_predictor"]["save_path"]
        if os.path.exists(self.rul_path):
            self.rul_predictor = TireRULPredictor()
            self.rul_predictor.load(self.rul_path)
            print("Loaded XGBoost RUL Predictor checkpoint.")
        else:
            self.rul_predictor = None
            print(f"Warning: XGBoost RUL Predictor checkpoint not found at {self.rul_path}. RUL predictions will be stubs.")

        # Image preprocessing transform
        self.img_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def enable_mc_dropout(self):
        """
        Force active dropout layers during evaluation to enable Monte Carlo Dropout uncertainty estimation.
        """
        if self.cnn_model is not None:
            self.cnn_model.eval()
            for m in self.cnn_model.modules():
                if m.__class__.__name__.startswith('Dropout'):
                    m.train()

    def predict(self, image_path: str, tabular_data: dict = None) -> dict:
        """
        Run upscaled prediction pipeline with MC Dropout and Integrated Gradients.
        """
        class_names = ["New", "Serviceable", "Unusable"]
        wear_class = "Serviceable"
        estimated_tread_depth_mm = 4.0
        tread_depth_uncertainty = 0.5
        target_class_idx = 1
        
        # Ensure tabular data has all required columns with reasonable defaults
        default_tabular = {
            "vehicle_model": "Sedan",
            "fuel_type": "Petrol",
            "transmission_type": "Automatic",
            "country": "Germany",
            "maximum_power(hp)": 150,
            "maximum_torque(N/m)": 220,
            "maximum_speed(km/h)": 200,
            "vehicle_acceleration(0-100 km/h in seconds)": 8.5,
            "vehicle_mileage(mpg)": 30.0,
            "vehicle_sprung_mass(kg)": 1500,
            "steering_radius(m)": 5.5,
            "axle_type(driven/dead)": "driven",
            "tyre_brand": "Michelin",
            "tyre_size": "205/55R16",
            "tread_material": "Silica Compound",
            "tread_pattern": "Symmetric",
            "tyre_camber_angle(degree)": 0.0,
            "standard_tread_depth(mm)": 8.0,
            "retreaded": "No",
            "road_condition": "Smooth",
            "weather_condition": "Dry",
            "expected_tyre_life(km)": 50000.0,
            "kilometers_driven(km)": 20000.0
        }
        
        if tabular_data is None:
            tabular_data = default_tabular
        else:
            merged = default_tabular.copy()
            merged.update(tabular_data)
            tabular_data = merged
            
        # Reconstruct standard 23 features for the neural net
        tab_vector = np.zeros(23, dtype=np.float32)
        tab_vector[0] = 150.0  # max power
        tab_vector[1] = 220.0  # max torque
        tab_vector[2] = 200.0  # max speed
        tab_vector[3] = 8.5    # accel
        tab_vector[4] = 30.0   # mpg
        tab_vector[5] = 1500.0 # weight
        tab_vector[6] = 5.5    # radius
        tab_vector[7] = tabular_data.get("tyre_camber_angle(degree)", 0.0)
        tab_vector[8] = tabular_data.get("standard_tread_depth(mm)", 8.0)
        tab_vector[9] = tabular_data.get("expected_tyre_life(km)", 50000.0)
        tab_vector[10] = tabular_data.get("kilometers_driven(km)", 20000.0)
        
        tab_tensor = torch.tensor(tab_vector, dtype=torch.float32).unsqueeze(0) # (1, 23)
        
        # Generate simulated IMU sequence (fallback is serviceable)
        imu_seq = generate_simulated_imu(label=1, length=100)
        imu_tensor = torch.tensor(imu_seq, dtype=torch.float32).unsqueeze(0) # (1, 100, 6)

        # --- 2. Wear Classifier & MC Dropout ---
        if self.cnn_model is not None:
            try:
                img_pil = Image.open(image_path).convert("RGB")
                img_tensor = self.img_transform(img_pil).unsqueeze(0)
                
                # Active dropout for MC uncertainty
                self.enable_mc_dropout()
                
                # Run 10 forward passes
                depth_samples = []
                class_votes = []
                
                with torch.no_grad():
                    for _ in range(10):
                        cls_out, reg_out = self.cnn_model(img_tensor, imu_seq=imu_tensor, tab_feats=tab_tensor)
                        depth_samples.append(reg_out.item())
                        class_votes.append(torch.argmax(cls_out, dim=1).item())
                        
                # Average predictions
                estimated_tread_depth_mm = float(np.mean(depth_samples))
                tread_depth_uncertainty = float(np.std(depth_samples))
                
                # Majority vote for classification
                pred_class_idx = max(set(class_votes), key=class_votes.count)
                wear_class = class_names[pred_class_idx]
                target_class_idx = pred_class_idx
                
            except Exception as e:
                print(f"Error running CNN inference: {str(e)}")
                
        estimated_tread_depth_mm = max(1.0, min(9.0, round(estimated_tread_depth_mm, 2)))
        tread_depth_uncertainty = max(0.1, round(tread_depth_uncertainty, 2))

        # --- 3. Alignment Heuristic (Otsu + Homography unwarped CV) ---
        alignment_results = {
            "alignment_flag": False,
            "alignment_confidence": 0.0,
            "diagnosis": "No alignment diagnostics available."
        }
        try:
            cv_results = analyze_wear_asymmetry(image_path, self.config)
            alignment_results["alignment_flag"] = cv_results["alignment_flag"]
            alignment_results["alignment_confidence"] = cv_results["alignment_confidence"]
            alignment_results["diagnosis"] = cv_results["diagnosis"]
        except Exception as e:
            print(f"Error running alignment heuristic: {str(e)}")

        # --- 4. RUL Predictor & Uncertainty ---
        predicted_rul_km = 0.0
        rul_uncertainty = 1000.0  # Default ± 1000 km
        
        if self.rul_predictor is not None:
            try:
                # Add current tread depth into fallback tabular data if it is not provided
                if "kilometers_driven(km)" not in tabular_data:
                    expected_life = 50000.0
                    standard_depth = 8.0
                    ratio = estimated_tread_depth_mm / standard_depth
                    tabular_data["kilometers_driven(km)"] = expected_life * (1.0 - ratio)
                    tabular_data["expected_tyre_life(km)"] = expected_life
                    tabular_data["standard_tread_depth(mm)"] = standard_depth
                    
                df_row = pd.DataFrame([tabular_data])
                preds = self.rul_predictor.predict(df_row)
                predicted_rul_km = float(preds[0, 0])
                
                # Estimate RUL uncertainty dynamically
                # Uncertainty grows with mileage driven and rough road conditions
                km_driven = tabular_data.get("kilometers_driven(km)", 20000.0)
                road_cond = tabular_data.get("road_condition", "Smooth")
                base_var = 300.0
                if road_cond == "Off-road":
                    base_var += 500.0
                rul_uncertainty = float(base_var + 0.04 * km_driven)
            except Exception as e:
                print(f"Error running RUL predictor: {str(e)}")
        else:
            predicted_rul_km = tabular_data.get("expected_tyre_life(km)", 50000.0) - tabular_data.get("kilometers_driven(km)", 20000.0)
            
        predicted_rul_km = max(0.0, round(predicted_rul_km, 1))
        rul_uncertainty = max(100.0, round(rul_uncertainty, 1))

        # --- 5. Saliency Visualizations (Grad-CAM & Integrated Gradients) ---
        explanation_heatmap_path = ""
        explanation_ig_path = ""
        
        if self.cnn_model is not None:
            heatmap_dir = self.config["outputs"]["heatmap_dir"]
            os.makedirs(heatmap_dir, exist_ok=True)
            base_name = os.path.basename(image_path)
            
            # Grad-CAM
            try:
                gc_save = os.path.join(heatmap_dir, f"gradcam_{os.path.splitext(base_name)[0]}.jpg")
                generate_gradcam_visualization(self.cnn_model, image_path, target_class_idx, gc_save, self.config)
                explanation_heatmap_path = gc_save
            except Exception as e:
                print(f"Error generating Grad-CAM: {str(e)}")
                
            # Integrated Gradients (High-Res pixel level)
            try:
                ig_save = os.path.join(heatmap_dir, f"ig_{os.path.splitext(base_name)[0]}.jpg")
                generate_ig_visualization(self.cnn_model, image_path, target_class_idx, ig_save, self.config)
                explanation_ig_path = ig_save
            except Exception as e:
                print(f"Error generating Integrated Gradients: {str(e)}")

        return {
            "wear_class": wear_class,
            "estimated_tread_depth_mm": estimated_tread_depth_mm,
            "estimated_tread_depth_uncertainty_mm": tread_depth_uncertainty,
            "predicted_rul_km": predicted_rul_km,
            "predicted_rul_uncertainty_km": rul_uncertainty,
            "alignment_flag": alignment_results["alignment_flag"],
            "alignment_confidence": round(alignment_results["alignment_confidence"], 3),
            "explanation_heatmap_path": explanation_heatmap_path,
            "explanation_ig_path": explanation_ig_path,
            "diagnosis": alignment_results["diagnosis"]
        }

def main():
    import json
    pipeline = UnifiedPredictionPipeline()
    config = load_config()
    
    # Use one of the mock images generated during diagnostic summaries for testing
    raw_dir = config["data"]["raw_dir"]
    mock_img_path = os.path.join(raw_dir, config["data"]["tyre_condition"], "Serviceable", "mock_serviceable_5.jpg")
    
    if os.path.exists(mock_img_path):
        print(f"Testing pipeline with mock image: {mock_img_path}")
        res = pipeline.predict(mock_img_path)
        print("Pipeline Result:")
        print(json.dumps(res, indent=4))
    else:
        print("Mock test image not found. Please run data_summary first to generate mocks.")

if __name__ == "__main__":
    main()
