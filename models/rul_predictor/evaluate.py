import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from utils.helper import load_config
from data.data_loader import load_rul_df
from models.rul_predictor.model import TireRULPredictor

def evaluate_rul_model():
    """
    Evaluate the trained RUL predictor model.
    Saves metrics report and comparison plots to outputs/
    """
    config = load_config()
    
    # Load DataFrame
    df = load_rul_df(config)
    
    # Apply subsampling exactly like train.py
    subsample_size = config["data"].get("rul_subsample_size", 100000)
    if subsample_size > 0 and len(df) > subsample_size:
        df = df.sample(n=subsample_size, random_state=config["general"]["seed"]).reset_index(drop=True)
        
    # Split features and targets
    targets = ["remaining_useful_life(km)", "current_tread_depth(mm)"]
    X = df.drop(columns=targets)
    y = df[targets]
    
    # Split
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=config["general"]["seed"]
    )
    
    # Load model
    save_path = config["models"]["rul_predictor"]["save_path"]
    if not os.path.exists(save_path):
        raise FileNotFoundError(f"Trained RUL model not found at {save_path}. Run train.py first.")
        
    predictor = TireRULPredictor()
    predictor.load(save_path)
    
    # Inference
    preds = predictor.predict(X_test)
    
    # Split targets
    y_test_rul = y_test["remaining_useful_life(km)"].values
    y_test_depth = y_test["current_tread_depth(mm)"].values
    
    preds_rul = preds[:, 0]
    preds_depth = preds[:, 1]
    
    # Metrics
    r2_rul = r2_score(y_test_rul, preds_rul)
    mae_rul = mean_absolute_error(y_test_rul, preds_rul)
    rmse_rul = np.sqrt(mean_squared_error(y_test_rul, preds_rul))
    
    r2_depth = r2_score(y_test_depth, preds_depth)
    mae_depth = mean_absolute_error(y_test_depth, preds_depth)
    rmse_depth = np.sqrt(mean_squared_error(y_test_depth, preds_depth))
    
    # Save Report
    output_dir = config["outputs"]["output_dir"]
    report_path = os.path.join(output_dir, "rul_predictor_report.txt")
    os.makedirs(output_dir, exist_ok=True)
    
    with open(report_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("           TABULAR RUL PREDICTOR EVALUATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write("1. REMAINING USEFUL LIFE (RUL) METRICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Mean Absolute Error (MAE) : {mae_rul:.2f} km\n")
        f.write(f"Root Mean Sq. Error (RMSE): {rmse_rul:.2f} km\n")
        f.write(f"R-squared Score (R2)      : {r2_rul:.4f}\n\n")
        
        f.write("2. TABULAR TREAD DEPTH ESTIMATE METRICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Mean Absolute Error (MAE) : {mae_depth:.4f} mm\n")
        f.write(f"Root Mean Sq. Error (RMSE): {rmse_depth:.4f} mm\n")
        f.write(f"R-squared Score (R2)      : {r2_depth:.4f}\n")
        
    print(f"\nSaved text evaluation report to: {report_path}")
    print("\n" + "=" * 60)
    print("           TABULAR RUL PREDICTOR EVALUATION SUMMARY")
    print("=" * 60)
    print(f"RUL Target (km)      -> MAE: {mae_rul:.2f}km, RMSE: {rmse_rul:.2f}km, R2: {r2_rul:.3f}")
    print(f"Tread Depth (mm)     -> MAE: {mae_depth:.2f}mm, RMSE: {rmse_depth:.2f}mm, R2: {r2_depth:.3f}")
    print("=" * 60 + "\n")
    
    # Plot 1: RUL Actual vs Predicted
    plt.figure(figsize=(6, 5))
    plt.scatter(y_test_rul, preds_rul, alpha=0.5, color="teal")
    lims = [min(y_test_rul.min(), preds_rul.min()), max(y_test_rul.max(), preds_rul.max())]
    plt.plot(lims, lims, "r--", alpha=0.75)
    plt.title("RUL Prediction: Predicted vs Actual")
    plt.xlabel("True Remaining Useful Life (km)")
    plt.ylabel("Predicted Remaining Useful Life (km)")
    plt.grid(True)
    plt.tight_layout()
    rul_plot_path = os.path.join(output_dir, "rul_prediction_scatter.png")
    plt.savefig(rul_plot_path)
    plt.close()
    print(f"Saved RUL regression plot to: {rul_plot_path}")
    
    # Plot 2: Tread Depth Actual vs Predicted
    plt.figure(figsize=(6, 5))
    plt.scatter(y_test_depth, preds_depth, alpha=0.5, color="orange")
    lims = [min(y_test_depth.min(), preds_depth.min()), max(y_test_depth.max(), preds_depth.max())]
    plt.plot(lims, lims, "r--", alpha=0.75)
    plt.title("Tabular Tread Depth: Predicted vs Actual")
    plt.xlabel("True Tread Depth (mm)")
    plt.ylabel("Predicted Tread Depth (mm)")
    plt.grid(True)
    plt.tight_layout()
    depth_plot_path = os.path.join(output_dir, "rul_depth_prediction_scatter.png")
    plt.savefig(depth_plot_path)
    plt.close()
    print(f"Saved Tread Depth regression plot to: {depth_plot_path}")

if __name__ == "__main__":
    evaluate_rul_model()
