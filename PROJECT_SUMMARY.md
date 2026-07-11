# Project Summary: Fused Tire Diagnostics & Metrology System

This document provides an easy-to-understand, high-level overview of what this project does, how it works, and the technical developments implemented.

---

## 🚗 Project Overview: What is this?
This project is an **automated tire health monitoring system**. It helps drivers and mechanics answer three vital questions:
1.  **How worn is my tire?** (Tread depth and wear severity class)
2.  **How long will it last?** (Remaining Useful Life in kilometers)
3.  **Is my wheel alignment incorrect?** (Camber/Toe misalignment detection)

Instead of relying only on a mechanic's visual check, this system **fuses two types of data**:
*   **Visual Data**: A close-up photograph of the tire tread.
*   **Telemetry/Sensor Data**: Dynamic vehicle metrics (mileage, typical road surfaces, and high-frequency IMU vibration signals).

---

## 🛠️ How It Works: The Core Components

The system connects four separate technologies into a unified pipeline:

```
[ Tire Crop Image ] ───► Homographic Unwarper ───► Left/Right Edge Density ───► Alignment Status (Camber & Toe)
[ Tire Crop Image ] ───► ResNet-18 Encoder ─────┐
[ Tabular Telemetry ] ──► Linear Projection ────┼─► Cross-Attention Fusion ──► Wear Class (New/Serviceable/Unusable)
[ IMU Vibration ] ─────► 1D CNN + Bi-LSTM ──────┘                              Tread Depth (mm)
[ Vehicle Mileage ] ───► XGBoost Tabular Regressor ──────────────────────────► Remaining Useful Life (RUL in km)
```

### 1. The Wear Classifier (Multimodal Deep Learning)
*   **What it does**: Takes the tire photo, the IMU vibration sequence (sensor readings from wheel hubs), and vehicle mileage telemetry, and outputs:
    *   *Wear Severity Class*: New, Serviceable, or Unusable.
    *   *Estimated Tread Depth*: Continuously estimated depth in millimeters (e.g., `3.76 mm`).
*   **The Novelty**: It uses **Cross-Attention Multihead Fusion**. The image features act as a Query to search through the sensor context (Key & Value), combining visual clues with vehicle history.

### 2. Knowledge Distillation (Smart Mobile Deployments)
*   Because high-frequency IMU vibrations aren't always available when taking a photo with a smartphone, we train a heavy **Teacher model** (Image + IMU + Tabular) and transfer its knowledge into a lightweight **Student model** (MobileNetV3-Small) that accepts **only images**.
*   This allows the mobile app to run fast on local devices while still benefiting from the multimodal patterns learned during training.

### 3. Wheel Alignment Heuristics (Classical Computer Vision)
*   **What it does**: Checks if your wheels are straight or misaligned.
*   **How it works**:
    1.  **Otsu Binarization**: Finds the outline of the tire.
    2.  **Homographic Unwarping**: Flatten/straightens the curved tread perspective into a flat rectangle. This removes perspective bias (caused by how you held the camera).
    3.  **Edge Density Check**: Divides the flat tread into Left, Center, and Right zones and counts the edge lines. If the Left or Right zone has fewer edges (meaning it is more worn/bald), it flags wheel misalignment.

### 4. Remaining Useful Life (RUL) Predictor
*   Takes vehicle parameters and current wear states and passes them to an **XGBoost regressor** to forecast how many kilometers of life the tire has left before reaching the wear limit.

### 5. Explainable AI (XAI)
*   Generates heatmaps using **Grad-CAM** (coarse model attention) and **Integrated Gradients** (pixel-level attributions) to highlight exactly which cracks, grooves, or balding areas influenced the AI's decision.

---

## 📈 Recent Developments (What We Implemented)

We built the following enhancements to make the prototype a premium diagnostic suite:

1.  **Physics-Guided Loss Constraint**: Updated the AI training loop to enforce a physical rule: *predicted tread depth must decrease as mileage increases*. Pairwise monotonicity checks prevent physically impossible depth estimations.
2.  **Quantitative Alignment Metrology**: Upgraded the alignment checker to compute actual numerical angles—**Camber Angle (in degrees)** and **Toe Deviation (in mm)**—by assessing the edge density slope across the unwarped tread.
3.  **Tire Size Dimension Metrology**: Created a dimensional parser that calculates overall tire diameter, sidewall height, and circumference from standard strings (like `205/55R16`).
4.  **Tire Fitment Safety Validator**: Checks if the selected tire size matches the vehicle class (e.g., raising safety alerts if a light Hatchback tire profile is selected on a heavy SUV/Truck).
5.  **Camber & Toe Gauges**: Created premium, color-coded horizontal slide indicators (Green/Yellow/Red zones) on the Streamlit dashboard, simulating a professional alignment readout.
6.  **CV Showroom (Homography & Canny)**: Displayed the intermediate flattened tread crop next to the Canny edge grid, letting users watch the perspective correction in real time.
7.  **Explainability Tabs**: Added side-by-side tabs comparing Grad-CAM and high-res Integrated Gradients attribution overlays.

---

## 📂 Key Code References

*   [dashboard/app.py](file:///Users/manu/python/Capstonetire/dashboard/app.py): Streamlit control dashboard containing the gauges, fitment validation, and CV showrooms.
*   [models/alignment_heuristic/analyzer.py](file:///Users/manu/python/Capstonetire/models/alignment_heuristic/analyzer.py): Otsu contour, Homographic unwarping, and Camber/Toe Metrology.
*   [models/wear_classifier/train.py](file:///Users/manu/python/Capstonetire/models/wear_classifier/train.py): Training routine including the Physics-Guided Monotonicity loss.
*   [pipeline/predict_pipeline.py](file:///Users/manu/python/Capstonetire/pipeline/predict_pipeline.py): Orchestrates images, tabular metadata, and outputs metrics in JSON format.
*   [explainability/integrated_gradients.py](file:///Users/manu/python/Capstonetire/explainability/integrated_gradients.py): Pixel-level explanation maps.
