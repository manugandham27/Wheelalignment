import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, r2_score, mean_absolute_error, mean_squared_error

from utils.helper import load_config, get_device
from data.data_loader import load_tyre_condition_images
from models.wear_classifier.dataset import TireWearDataset
from models.wear_classifier.model import DualHeadTireCNN

def evaluate_model():
    """
    Evaluate the trained upscaled multimodal CNN model.
    """
    config = load_config()
    device = get_device(config)
    
    # Load paths and split exactly like train.py using the seed
    image_paths, labels = load_tyre_condition_images(config)
    if len(image_paths) == 0:
        raise ValueError("No images found for evaluation.")

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=config["general"]["seed"], stratify=labels
    )
    
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    weak_label_cfg = config["models"]["wear_classifier"]["tread_depth_weak_labels"]
    val_dataset = TireWearDataset(val_paths, val_labels, transform=val_transforms, weak_label_cfg=weak_label_cfg)
    val_loader = DataLoader(val_dataset, batch_size=config["models"]["wear_classifier"]["batch_size"], shuffle=False)
    
    # Instantiate and load model
    backbone = config["models"]["wear_classifier"].get("backbone", "resnet18")
    num_classes = config["models"]["wear_classifier"]["num_classes"]
    save_path = config["models"]["wear_classifier"]["save_path"]
    
    if not os.path.exists(save_path):
        raise FileNotFoundError(f"Trained model checkpoint not found at {save_path}. Please run train.py first.")
        
    model = DualHeadTireCNN(backbone_name=backbone, pretrained=False, num_classes=num_classes)
    model.load_state_dict(torch.load(save_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Accumulators
    all_preds_cls = []
    all_targets_cls = []
    all_preds_reg = []
    all_targets_reg = []
    
    print("Running evaluation on validation set...")
    with torch.no_grad():
        for imgs, cls_lbls, reg_depths, imu_seqs, tab_feats in val_loader:
            imgs = imgs.to(device)
            imu_seqs = imu_seqs.to(device)
            tab_feats = tab_feats.to(device)
            
            cls_out, reg_out = model(imgs, imu_seq=imu_seqs, tab_feats=tab_feats)
            
            _, predicted = torch.max(cls_out, 1)
            
            all_preds_cls.extend(predicted.cpu().numpy())
            all_targets_cls.extend(cls_lbls.numpy())
            all_preds_reg.extend(reg_out.cpu().numpy())
            all_targets_reg.extend(reg_depths.numpy())
            
    # Convert lists to arrays
    all_preds_cls = np.array(all_preds_cls)
    all_targets_cls = np.array(all_targets_cls)
    all_preds_reg = np.array(all_preds_reg)
    all_targets_reg = np.array(all_targets_reg)
    
    class_names = ["New", "Serviceable", "Unusable"]
    
    # 1. Classification Metrics
    cls_report = classification_report(all_targets_cls, all_preds_cls, target_names=class_names)
    conf_mat = confusion_matrix(all_targets_cls, all_preds_cls)
    
    # 2. Regression Metrics
    r2 = r2_score(all_targets_reg, all_preds_reg)
    mae = mean_absolute_error(all_targets_reg, all_preds_reg)
    rmse = np.sqrt(mean_squared_error(all_targets_reg, all_preds_reg))
    
    # Save Report
    output_dir = config["outputs"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "wear_classifier_report.txt")
    
    with open(report_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("      WEAR CLASSIFIER & TREAD DEPTH REGRESSION EVALUATION REPORT (UPSCALED FUSION)\n")
        f.write("=" * 60 + "\n\n")
        f.write("1. CLASSIFICATION METRICS (Wear Severity)\n")
        f.write("-" * 40 + "\n")
        f.write(cls_report)
        f.write("\n")
        f.write("2. REGRESSION METRICS (Continuous Tread Depth in mm)\n")
        f.write("-" * 40 + "\n")
        f.write(f"Mean Absolute Error (MAE) : {mae:.4f} mm\n")
        f.write(f"Root Mean Sq. Error (RMSE): {rmse:.4f} mm\n")
        f.write(f"R-squared Score (R2)      : {r2:.4f}\n")
        
    print(f"\nSaved text evaluation report to: {report_path}")
    print("\n" + "=" * 60)
    print("      WEAR CLASSIFIER & TREAD DEPTH REGRESSION EVALUATION SUMMARY (UPSCALED FUSION)")
    print("=" * 60)
    print(cls_report)
    print("-" * 60)
    print(f"Tread Depth Regression -> MAE: {mae:.2f}mm, RMSE: {rmse:.2f}mm, R2: {r2:.3f}")
    print("=" * 60 + "\n")
    
    # Generate and Save Confusion Matrix Plot
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Wear Classifier Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    conf_mat_path = os.path.join(output_dir, "wear_classifier_confusion_matrix.png")
    plt.savefig(conf_mat_path)
    plt.close()
    
    # Generate and Save Regressor Scatter Plot
    plt.figure(figsize=(6, 5))
    plt.scatter(all_targets_reg, all_preds_reg, alpha=0.6, color="purple")
    lims = [
        np.min([all_targets_reg.min(), all_preds_reg.min()]),
        np.max([all_targets_reg.max(), all_preds_reg.max()]),
    ]
    plt.plot(lims, lims, "r--", alpha=0.75, zorder=0)
    plt.title("Tread Depth Estimation: Predicted vs Actual")
    plt.xlabel("True Tread Depth (mm)")
    plt.ylabel("Predicted Tread Depth (mm)")
    plt.grid(True)
    plt.tight_layout()
    scatter_path = os.path.join(output_dir, "wear_classifier_regression_scatter.png")
    plt.savefig(scatter_path)
    plt.close()

if __name__ == "__main__":
    evaluate_model()
