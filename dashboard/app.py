import os
import sys
# Inject project root path into Python path to resolve imports properly when run via Streamlit
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from PIL import Image
import pandas as pd
import json

from utils.helper import load_config
from pipeline.predict_pipeline import UnifiedPredictionPipeline
import numpy as np

def calculate_tire_dimensions(size_str):
    """
    Calculate physical tire dimensions (width, sidewall height, diameter, circumference)
    from standard size strings like '205/55R16'.
    """
    try:
        parts = size_str.split('/')
        width = float(parts[0])
        aspect_ratio = float(parts[1].split('R')[0])
        rim_diameter_inch = float(parts[1].split('R')[1])
        
        sidewall_height_mm = width * (aspect_ratio / 100.0)
        rim_diameter_mm = rim_diameter_inch * 25.4
        total_diameter_mm = rim_diameter_mm + 2 * sidewall_height_mm
        total_diameter_inch = total_diameter_mm / 25.4
        circumference_mm = np.pi * total_diameter_mm
        
        return {
            "width": width,
            "aspect_ratio": aspect_ratio,
            "rim_diameter": rim_diameter_inch,
            "sidewall_height": sidewall_height_mm,
            "total_diameter_mm": total_diameter_mm,
            "total_diameter_inch": total_diameter_inch,
            "circumference_mm": circumference_mm
        }
    except Exception:
        return None

def render_alignment_gauge(label, val, min_val, max_val, unit, target_val=0.0):
    """
    Renders a premium visual alignment progress gauge with colored safety boundaries (Green/Yellow/Red).
    """
    # Normalized position from 0% to 100%
    pct = (val - min_val) / (max_val - min_val) * 100
    pct = max(0, min(100, pct))
    
    # Determine color zone
    abs_val = abs(val - target_val)
    if abs_val <= 0.5:
        color = "#2ecc71" # Safe green
        status = "Optimal"
    elif abs_val <= 1.2:
        color = "#f1c40f" # Alert yellow
        status = "Borderline"
    else:
        color = "#e74c3c" # Warning red
        status = "Critical Misalignment"
        
    html = f"""
    <div style="margin-bottom:15px; padding:12px; background:rgba(255,255,255,0.02); border-radius:8px; border:1px solid rgba(255,255,255,0.05);">
        <div style="display:flex; justify-content:space-between; font-size:0.85rem; color:#b0b3c0; font-weight:600; margin-bottom:4px;">
            <span>{label}</span>
            <span style="color:{color};">{val:+.2f} {unit} ({status})</span>
        </div>
        <div style="position:relative; width:100%; height:8px; background:rgba(255,255,255,0.1); border-radius:4px; margin:6px 0;">
            <!-- Target marker (0.0) -->
            <div style="position:absolute; left:50%; top:-2px; width:2px; height:12px; background:#fff; z-index:2;"></div>
            <!-- Value marker -->
            <div style="position:absolute; left:{pct}%; top:-4px; width:10px; height:16px; background:{color}; border-radius:2px; border:1px solid #fff; transform:translateX(-5px); z-index:3;"></div>
            <!-- Zones background -->
            <div style="position:absolute; left:0; top:0; width:100%; height:100%; display:flex; border-radius:4px; overflow:hidden; opacity:0.25;">
                <div style="width:30%; height:100%; background:#e74c3c;"></div>
                <div style="width:15%; height:100%; background:#f1c40f;"></div>
                <div style="width:10%; height:100%; background:#2ecc71;"></div>
                <div style="width:15%; height:100%; background:#f1c40f;"></div>
                <div style="width:30%; height:100%; background:#e74c3c;"></div>
            </div>
        </div>
        <div style="display:flex; justify-content:space-between; font-size:0.7rem; color:#686c75;">
            <span>{min_val:+.1f}</span>
            <span>0.0 (Target)</span>
            <span>{max_val:+.1f}</span>
        </div>
    </div>
    """
    return html


# Config page details
st.set_page_config(
    page_title="Fused Tire Diagnostics Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling via CSS (Modern Sleek Dark Mode Theme)
st.markdown("""
<style>
    .main {
        background-color: #0f1116;
        color: #f0f2f6;
    }
    .stApp {
        background-color: #0f1116;
    }
    h1, h2, h3 {
        color: #ffffff !important;
        font-family: 'Outfit', 'Inter', sans-serif;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 15px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.35);
        backdrop-filter: blur(10px);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(255, 255, 255, 0.15);
    }
    .status-ok {
        border-left: 5px solid #2ecc71 !important;
    }
    .status-warn {
        border-left: 5px solid #f1c40f !important;
    }
    .status-danger {
        border-left: 5px solid #e74c3c !important;
    }
    .card-label {
        font-size: 0.85rem;
        color: #8a8d98;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
        font-weight: 600;
    }
    .card-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #ffffff;
    }
    .card-desc {
        font-size: 0.85rem;
        color: #b0b3c0;
        margin-top: 5px;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- SESSION STATE & INITIALIZATION -----------------
@st.cache_resource
def get_pipeline():
    """
    Initialize and cache pipeline to prevent re-instantiation overhead on interaction.
    """
    return UnifiedPredictionPipeline()

try:
    pipeline = get_pipeline()
    config = pipeline.config
except Exception as e:
    st.error(f"Error loading prediction pipeline: {str(e)}")
    st.stop()

# ----------------- SIDEBAR INPUTS -----------------
st.sidebar.title("🛠️ Diagnostics Controls")
st.sidebar.write("Configure input settings for the tabular RUL predictor and CV alignment heuristics.")

# Common preset selector
scenario = st.sidebar.selectbox(
    "Select Preset Vehicle Profile",
    ["Custom Manual Entry", "New Tire Baseline", "Over-inflated City Commute", "Under-inflated Highway Driving", "Rough Off-Road Terrain"]
)

# Standard preset mappings
preset_vals = {
    "New Tire Baseline": {
        "expected_life": 60000,
        "km_driven": 2000,
        "camber": 0.0,
        "road_condition": "Smooth",
        "weather_condition": "Dry",
        "brand": "Michelin",
        "size": "205/55R16",
        "retreaded": "No"
    },
    "Over-inflated City Commute": {
        "expected_life": 50000,
        "km_driven": 15000,
        "camber": -0.2,
        "road_condition": "Smooth",
        "weather_condition": "Humid",
        "brand": "Continental",
        "size": "225/65R17",
        "retreaded": "No"
    },
    "Under-inflated Highway Driving": {
        "expected_life": 55000,
        "km_driven": 35000,
        "camber": 0.5,
        "road_condition": "Smooth",
        "weather_condition": "Cold",
        "brand": "Bridgestone",
        "size": "245/40R18",
        "retreaded": "No"
    },
    "Rough Off-Road Terrain": {
        "expected_life": 40000,
        "km_driven": 18000,
        "camber": -1.5,
        "road_condition": "Off-road",
        "weather_condition": "Rainy",
        "brand": "Goodyear",
        "size": "195/65R15",
        "retreaded": "Yes"
    }
}

# Determine values
if scenario != "Custom Manual Entry":
    p = preset_vals[scenario]
    st.sidebar.info(f"Loaded preset parameters for '{scenario}'. Settings are adjustable below.")
else:
    p = {
        "expected_life": 50000,
        "km_driven": 20000,
        "camber": 0.0,
        "road_condition": "Smooth",
        "weather_condition": "Dry",
        "brand": "Michelin",
        "size": "205/55R16",
        "retreaded": "No"
    }

# Render input form
with st.sidebar.form("vehicle_params_form"):
    st.markdown("### 🛞 Sensor Readings")
    
    brand = st.selectbox("Tire Brand", ["Michelin", "Bridgestone", "Continental", "Goodyear", "Pirelli"], index=["Michelin", "Bridgestone", "Continental", "Goodyear", "Pirelli"].index(p["brand"]))
    size = st.selectbox("Tire Size Profile", ["205/55R16", "225/65R17", "245/40R18", "195/65R15"], index=["205/55R16", "225/65R17", "245/40R18", "195/65R15"].index(p["size"]))
    
    expected_life = st.slider("Expected Tire Lifespan (km)", 30000, 80000, p["expected_life"], step=5000)
    km_driven = st.slider("Current Distance Driven (km)", 0, int(expected_life), p["km_driven"], step=1000)
    camber = st.slider("Camber Alignment Angle (deg)", -4.0, 4.0, p["camber"], step=0.1)
    
    st.markdown("### 🗺️ Environmental Conditions")
    road_cond = st.selectbox("Typical Road Surface", ["Smooth", "Rough", "Off-road"], index=["Smooth", "Rough", "Off-road"].index(p["road_condition"]))
    weather_cond = st.selectbox("Dominant Weather", ["Dry", "Humid", "Cold", "Rainy"], index=["Dry", "Humid", "Cold", "Rainy"].index(p["weather_condition"]))
    retreaded = st.selectbox("Retreaded Tire Status", ["No", "Yes"], index=["No", "Yes"].index(p["retreaded"]))
    
    vehicle_model = st.selectbox("Vehicle Body Class", ["Sedan", "SUV", "Hatchback", "Truck", "Coupe"], index=0)
    
    # Hidden defaults or simplified parameters mapped to Kaggle schema
    fuel_type = "Petrol"
    transmission_type = "Automatic"
    country = "Germany"
    max_power = 150
    max_torque = 220
    max_speed = 200
    accel = 8.5
    mileage = 30.0
    sprung_mass = 1500
    steering_rad = 5.5
    axle_type = "driven"
    tread_material = "Silica Compound"
    tread_pattern = "Symmetric"
    standard_depth = 8.0
    
    submit_button = st.form_submit_button(label="Apply Parameter Changes")

# Compile sensor input dictionary
sensor_input = {
    "vehicle_model": vehicle_model,
    "fuel_type": fuel_type,
    "transmission_type": transmission_type,
    "country": country,
    "maximum_power(hp)": max_power,
    "maximum_torque(N/m)": max_torque,
    "maximum_speed(km/h)": max_speed,
    "vehicle_acceleration(0-100 km/h in seconds)": accel,
    "vehicle_mileage(mpg)": mileage,
    "vehicle_sprung_mass(kg)": sprung_mass,
    "steering_radius(m)": steering_rad,
    "axle_type(driven/dead)": axle_type,
    "tyre_brand": brand,
    "tyre_size": size,
    "tread_material": tread_material,
    "tread_pattern": tread_pattern,
    "tyre_camber_angle(degree)": camber,
    "standard_tread_depth(mm)": standard_depth,
    "retreaded": retreaded,
    "road_condition": road_cond,
    "weather_condition": weather_cond,
    "expected_tyre_life(km)": expected_life,
    "kilometers_driven(km)": km_driven
}

# ----------------- MAIN UI CONTENT -----------------
st.title("🛞 Fused Wear Estimation & Alignment Detection System")
st.write("continuous tire diagnostics prototype fusing deep vision representations with vehicle inertial data.")
st.markdown("---")

# Main page layout splitting image uploader and predictions
col1, col2 = st.columns([1, 1.2])

with col1:
    st.markdown("### 📷 Tread Image Upload")
    uploaded_file = st.file_uploader(
        "Upload a close-up tire face/tread photograph (JPG/PNG)", 
        type=["jpg", "jpeg", "png"],
        help="Upload a cropped tread face photograph. Ensure vertical alignment of treads for CV analytics."
    )
    
    # If no file uploaded, provide option to use a mock image from generated directories
    st.write("---")
    st.write("💡 *No tire photo? Select one of the pre-loaded testing mocks:*")
    
    raw_dir = config["data"]["raw_dir"]
    tyre_cond_dir = config["data"]["tyre_condition"]
    
    mock_files = []
    mock_base = os.path.join(raw_dir, tyre_cond_dir)
    if os.path.exists(mock_base):
        for root, _, files in os.walk(mock_base):
            for f in files:
                if f.lower().endswith(".jpg"):
                    mock_files.append(os.path.join(root, f))
                    
    selected_mock_path = None
    if mock_files:
        mock_display_names = [os.path.relpath(f, mock_base) for f in mock_files]
        selected_display = st.selectbox("Choose a mock image", ["None - Upload Custom Image"] + mock_display_names)
        if selected_display != "None - Upload Custom Image":
            selected_mock_path = mock_files[mock_display_names.index(selected_display)]
            
    # Load image for processing
    active_img_path = None
    temp_img_name = "outputs/temp_uploaded_tire.jpg"
    
    if uploaded_file is not None:
        # Save temporary uploaded file
        os.makedirs("outputs", exist_ok=True)
        with open(temp_img_name, "wb") as f:
            f.write(uploaded_file.getbuffer())
        active_img_path = temp_img_name
        st.success("Custom tire image loaded successfully!")
    elif selected_mock_path is not None:
        active_img_path = selected_mock_path
        st.success(f"Loaded test mock: `{os.path.basename(selected_mock_path)}`")
        
    if active_img_path:
        # Display image preview
        img = Image.open(active_img_path)
        st.image(img, caption="Loaded Input Tread Photograph", use_column_width=True)
        
        # --- Tire Dimension & Fitment Validator ---
        st.markdown("---")
        st.markdown("### 📐 Tire Dimension Metrology")
        dimensions = calculate_tire_dimensions(size)
        if dimensions:
            st.markdown(f"""
            <div class="metric-card" style="margin-bottom: 10px; padding: 15px;">
                <div style="font-weight:700; font-size:1.1rem; color:#fff; margin-bottom:5px;">Dimension Readings ({size})</div>
                <div style="font-size:0.85rem; color:#b0b3c0; line-height:1.5;">
                    • <b>Section Width:</b> {dimensions['width']:.0f} mm<br/>
                    • <b>Sidewall Height:</b> {dimensions['sidewall_height']:.1f} mm<br/>
                    • <b>Overall Diameter:</b> {dimensions['total_diameter_inch']:.2f} in ({dimensions['total_diameter_mm']:.1f} mm)<br/>
                    • <b>Circumference:</b> {dimensions['circumference_mm']:.1f} mm
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Compatibility Validation
            if vehicle_model in ["SUV", "Truck"] and size in ["195/65R15", "205/55R16"]:
                st.error(f"⚠️ **Fitment Alert:** Selected profile ({size}) has an insufficient load capacity for a heavy **{vehicle_model}**. Risk of rapid wear, sidewall collapse, or structural failure!")
            elif vehicle_model == "Hatchback" and size in ["245/40R18", "225/65R17"]:
                st.warning(f"⚠️ **Fitment Warning:** Selected profile ({size}) is oversized for a **Hatchback**. Risk of wheel well rubbing, steering binding, and severe speedometer error.")
            else:
                st.success(f"✅ **Fitment Status:** Compatible tire profile ({size}) for Selected Vehicle Class (**{vehicle_model}**).")
    else:
        st.warning("⚠️ Please upload a tire photo or select a mock image to run diagnostics.")

# ----------------- INFERENCE & RESULTS DISPLAY -----------------
with col2:
    st.markdown("### 📊 Diagnostics Results")
    
    if active_img_path:
        with st.spinner("Processing image and fusing inertial metrics..."):
            # Run prediction pipeline
            res = pipeline.predict(active_img_path, tabular_data=sensor_input)
            
        # Determine CSS class for severity color coding
        severity = res["wear_class"]
        if severity == "New":
            status_cls = "status-ok"
            badge_color = "#2ecc71"
        elif severity == "Serviceable":
            status_cls = "status-warn"
            badge_color = "#f1c40f"
        else:
            status_cls = "status-danger"
            badge_color = "#e74c3c"
            
        # Render color card metrics
        subcol1, subcol2 = st.columns(2)
        
        with subcol1:
            st.markdown(f"""
            <div class="metric-card {status_cls}">
                <div class="card-label">Wear Severity Classification</div>
                <div class="card-value" style="color:{badge_color};">{res['wear_class']}</div>
                <div class="card-desc">Calculated from ResNet-18 visual representations.</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Tread depth metric
            st.markdown(f"""
            <div class="metric-card">
                <div class="card-label">Estimated Tread Depth</div>
                <div class="card-value">{res['estimated_tread_depth_mm']:.2f} mm</div>
                <div class="card-desc">Standard new tire baseline is approx 8.0 mm. Minimum legal limit is 1.6 mm.</div>
            </div>
            """, unsafe_allow_html=True)
            
        with subcol2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="card-label">Remaining Useful Life (RUL)</div>
                <div class="card-value">{res['predicted_rul_km']:,.0f} km</div>
                <div class="card-desc">Fused estimate using XGBoost tabular model & visual tread depth representation.</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Alignment Status card
            align_cls = "status-danger" if res["alignment_flag"] else "status-ok"
            align_txt = "MISALIGNED" if res["alignment_flag"] else "ALIGNED"
            align_color = "#e74c3c" if res["alignment_flag"] else "#2ecc71"
            
            st.markdown(f"""
            <div class="metric-card {align_cls}">
                <div class="card-label">Wheel Alignment Status</div>
                <div class="card-value" style="color:{align_color};">{align_txt}</div>
                <div class="card-desc">Symmetry Confidence: {res['alignment_confidence']*100:.1f}% (threshold {config['models']['alignment_heuristic']['asymmetry_threshold'] * 100:.0f}%)</div>
            </div>
            """, unsafe_allow_html=True)

        # Show diagnostics notes
        st.info(f"🔎 **Diagnostic Insights:** {res['diagnosis']}")

        # Show Camber and Toe dials
        st.markdown("### 📊 Wheel Alignment Metrology")
        camber_html = render_alignment_gauge("Estimated Camber Angle", res["estimated_camber_deg"], -3.0, 3.0, "°")
        toe_html = render_alignment_gauge("Estimated Toe Deviation", res["estimated_toe_mm"], -5.0, 5.0, "mm")
        st.markdown(camber_html, unsafe_allow_html=True)
        st.markdown(toe_html, unsafe_allow_html=True)

        # Show TSA CV Showroom
        if res.get("unwarped_path") and os.path.exists(res["unwarped_path"]):
            st.markdown("### 🛞 Classical CV Tread Symmetry Analysis (TSA)")
            st.write("Perspective distortion correction using homographic unwarping and zonal Canny edge analysis:")
            subcol_cv1, subcol_cv2 = st.columns(2)
            with subcol_cv1:
                st.image(Image.open(res["unwarped_path"]), caption="Homographically Unwarped Tread (Perspective Corrected)", use_column_width=True)
            with subcol_cv2:
                st.image(Image.open(res["edges_path"]), caption="Zonal Canny Edges (Left / Center / Right Zones)", use_column_width=True)

        # Show Dual-Explainability Visualizer (Grad-CAM vs. Integrated Gradients)
        st.markdown("### 🔍 Explainability & Attention Analysis")
        st.write("Compare CNN visual attention regions (Grad-CAM) with pixel-level mathematical wear attributions (Integrated Gradients).")
        tab_gc, tab_ig = st.tabs(["Grad-CAM Saliency", "Integrated Gradients (High-Res)"])
        
        with tab_gc:
            if res.get("explanation_heatmap_path") and os.path.exists(res["explanation_heatmap_path"]):
                st.image(Image.open(res["explanation_heatmap_path"]), caption="Grad-CAM Wear Classifier Heatmap Overlay", use_column_width=True)
            else:
                st.info("No Grad-CAM explanation path available.")
                
        with tab_ig:
            if res.get("explanation_ig_path") and os.path.exists(res["explanation_ig_path"]):
                st.image(Image.open(res["explanation_ig_path"]), caption="Integrated Gradients (IG) Wear Saliency Map", use_column_width=True)
            else:
                st.info("No Integrated Gradients explanation path available.")
            
        # Display full raw JSON output for transparency
        st.write("---")
        with st.expander("📝 View Pipeline Raw JSON Output"):
            st.json(res)
    else:
        st.info("Results will appear here once an image and features are submitted.")
