import os
import csv
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from sklearn.metrics import mean_absolute_error, mean_squared_error

from utils.helper import load_config
from data.data_loader import generate_mock_image
from models.wear_classifier.model import DualHeadTireCNN

def check_and_generate_mock_holdout(config):
    """
    Check if a holdout validation dataset exists.
    If not, generate a mock holdout CSV and corresponding mock images for testing the evaluation module.
    """
    holdout_dir = "evaluation/holdout"
    manifest_path = "evaluation/holdout_manifest.csv"
    
    if os.path.exists(manifest_path):
        return manifest_path
        
    print("Holdout validation set not found. Generating mock holdout validation data...")
    os.makedirs(holdout_dir, exist_ok=True)
    
    # 20 holdout samples with various manually measured depths (ranging from 1.5mm to 8.2mm)
    holdout_data = [
        {"filename": "h_tire_0.jpg", "manual_tread_depth_mm": 8.0, "wear": "new"},
        {"filename": "h_tire_1.jpg", "manual_tread_depth_mm": 7.5, "wear": "new"},
        {"filename": "h_tire_2.jpg", "manual_tread_depth_mm": 6.8, "wear": "new"},
        {"filename": "h_tire_3.jpg", "manual_tread_depth_mm": 5.5, "wear": "serviceable"},
        {"filename": "h_tire_4.jpg", "manual_tread_depth_mm": 4.8, "wear": "serviceable"},
        {"filename": "h_tire_5.jpg", "manual_tread_depth_mm": 4.0, "wear": "serviceable"},
        {"filename": "h_tire_6.jpg", "manual_tread_depth_mm": 3.2, "wear": "serviceable"},
        {"filename": "h_tire_7.jpg", "manual_tread_depth_mm": 2.5, "wear": "serviceable"},
        {"filename": "h_tire_8.jpg", "manual_tread_depth_mm": 1.8, "wear": "unusable"},
        {"filename": "h_tire_9.jpg", "manual_tread_depth_mm": 1.5, "wear": "unusable"},
        {"filename": "h_tire_10.jpg", "manual_tread_depth_mm": 8.2, "wear": "new"},
        {"filename": "h_tire_11.jpg", "manual_tread_depth_mm": 7.1, "wear": "new"},
        {"filename": "h_tire_12.jpg", "manual_tread_depth_mm": 6.2, "wear": "new"},
        {"filename": "h_tire_13.jpg", "manual_tread_depth_mm": 5.0, "wear": "serviceable"},
        {"filename": "h_tire_14.jpg", "manual_tread_depth_mm": 4.4, "wear": "serviceable"},
        {"filename": "h_tire_15.jpg", "manual_tread_depth_mm": 3.8, "wear": "serviceable"},
        {"filename": "h_tire_16.jpg", "manual_tread_depth_mm": 2.9, "wear": "serviceable"},
        {"filename": "h_tire_17.jpg", "manual_tread_depth_mm": 2.1, "wear": "unusable"},
        {"filename": "h_tire_18.jpg", "manual_tread_depth_mm": 1.6, "wear": "unusable"},
        {"filename": "h_tire_19.jpg", "manual_tread_depth_mm": 1.2, "wear": "unusable"}
    ]
    
    # Save the manifest CSV
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "manual_tread_depth_mm"])
        writer.writeheader()
        for row in holdout_data:
            writer.writerow({"filename": row["filename"], "manual_tread_depth_mm": row["manual_tread_depth_mm"]})
            
            # Generate the mock image for this holdout row
            img_path = os.path.join(holdout_dir, row["filename"])
            generate_mock_image(img_path, wear_type=row["wear"])
            
    print(f"Generated mock holdout data: 20 images in {holdout_dir} and manifest {manifest_path}")
    return manifest_path

def run_holdout_validation():
    """
    Load validation CSV and run the trained CNN regression head to compare
    estimated vs manually measured tread depths.
    """
    config = load_config()
    
    # Check and generate validation mock data if missing
    manifest_path = check_and_generate_mock_holdout(config)
    holdout_dir = "evaluation/holdout"
    
    # Read manifest CSV
    df_manifest = pd.read_csv(manifest_path)
    
    # Load model
    cnn_path = config["models"]["wear_classifier"]["save_path"]
    if not os.path.exists(cnn_path):
        print(f"Warning: Trained CNN wear classifier not found at {cnn_path}. Can't run validation. Please train the model first.")
        return
        
    backbone = config["models"]["wear_classifier"]["backbone"]
    num_classes = config["models"]["wear_classifier"]["num_classes"]
    
    model = DualHeadTireCNN(backbone_name=backbone, pretrained=False, num_classes=num_classes)
    model.load_state_dict(torch.load(cnn_path, map_location=torch.device("cpu")))
    model.eval()
    
    img_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    results = []
    
    print("\nRunning Holdout Validation...")
    for idx, row in df_manifest.iterrows():
        img_name = row["filename"]
        manual_depth = float(row["manual_tread_depth_mm"])
        img_path = os.path.join(holdout_dir, img_name)
        
        if not os.path.exists(img_path):
            print(f"Warning: Holdout image {img_path} not found. Skipping.")
            continue
            
        try:
            img_pil = Image.open(img_path).convert("RGB")
            img_tensor = img_transform(img_pil).unsqueeze(0)
            
            with torch.no_grad():
                _, reg_out = model(img_tensor)
                estimated_depth = float(reg_out.item())
                
            error = estimated_depth - manual_depth
            results.append({
                "filename": img_name,
                "manual_depth_mm": manual_depth,
                "estimated_depth_mm": round(estimated_depth, 2),
                "error_mm": round(error, 2)
            })
        except Exception as e:
            print(f"Error validating {img_name}: {str(e)}")
            
    if not results:
        print("No holdout validation images were processed.")
        return
        
    df_results = pd.DataFrame(results)
    
    # Calculate errors
    mae = mean_absolute_error(df_results["manual_depth_mm"], df_results["estimated_depth_mm"])
    rmse = np.sqrt(mean_squared_error(df_results["manual_depth_mm"], df_results["estimated_depth_mm"]))
    mean_bias = df_results["error_mm"].mean()
    
    # Write report file
    output_dir = config["outputs"]["output_dir"]
    report_path = os.path.join(output_dir, "holdout_validation_report.txt")
    os.makedirs(output_dir, exist_ok=True)
    
    with open(report_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("            HOLDOUT TREAD-DEPTH VALIDATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write("1. PERFORMANCE STATISTICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total Test Samples        : {len(df_results)}\n")
        f.write(f"Mean Absolute Error (MAE) : {mae:.4f} mm\n")
        f.write(f"Root Mean Sq. Error (RMSE): {rmse:.4f} mm\n")
        f.write(f"Average Bias (Pred - True): {mean_bias:.4f} mm\n\n")
        
        f.write("2. SAMPLE-LEVEL RESULTS\n")
        f.write("-" * 40 + "\n")
        f.write(df_results.to_string(index=False))
        f.write("\n")
        
    print(df_results.to_string(index=False))
    print("-" * 60)
    print(f"Validation Holdout Complete -> MAE: {mae:.2f}mm, RMSE: {rmse:.2f}mm, Bias: {mean_bias:.2f}mm")
    print(f"Report saved to: {report_path}\n")

if __name__ == "__main__":
    run_holdout_validation()
