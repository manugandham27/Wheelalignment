import os
import glob
import cv2
import numpy as np
import pandas as pd
from utils.helper import load_config

def generate_simulated_imu(label, length=100, seed=None):
    """
    Generate simulated 6-axis time-series IMU data (shape: length x 6).
    Columns: [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]
    - New tire (label 0): stable periodic rotation, low vibration noise.
    - Serviceable (label 1): moderate periodic signal, medium noise.
    - Unusable (label 2): heavily degraded periodic signal, high vibration amplitude (high noise).
    """
    if seed is not None:
        np.random.seed(seed)
        
    t = np.linspace(0, 10, length)
    
    # Rotational frequency (periodic wheel rotation)
    f = 2.0  # 2 Hz rotation
    
    if label == 0:
        noise_level = 0.1
        vibration_amp = 1.0
    elif label == 1:
        noise_level = 0.3
        vibration_amp = 1.3
    else:  # label == 2
        noise_level = 0.7
        vibration_amp = 1.8

    # Simulate accelerometer readings (g-forces)
    accel_x = vibration_amp * np.sin(2 * np.pi * f * t) + np.random.normal(0, noise_level, length)
    accel_y = vibration_amp * np.cos(2 * np.pi * f * t) + np.random.normal(0, noise_level, length)
    accel_z = 9.81 + np.random.normal(0, noise_level, length)  # Gravity baseline
    
    # Simulate gyroscope readings (angular velocity in rad/s)
    gyro_x = np.random.normal(0, noise_level * 0.5, length)
    gyro_y = 12.5 + np.random.normal(0, noise_level * 0.8, length)  # Rotational velocity
    gyro_z = np.random.normal(0, noise_level * 0.5, length)
    
    imu_data = np.stack([accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z], axis=1) # (length, 6)
    return imu_data

def generate_mock_image(filepath, wear_type="new", asymmetry_side=None):
    """
    Generate a mock tire tread image using OpenCV and numpy.
    - new: Deep, clean treads (thick dark grooves and bright tread blocks).
    - serviceable: Partially worn treads (thinner grooves, less contrast).
    - unusable: Smooth, bald surface (nearly uniform gray, no distinct treads).
    - asymmetry_side: 'left' or 'right' to wear down that specific side (lower contrast/groove depth) for testing alignment.
    """
    # 224x224 grayscale-like RGB image
    img = np.ones((224, 224, 3), dtype=np.uint8) * 180  # Base gray tire color
    
    # Draw vertical tread bands (tread grooves)
    groove_color = (40, 40, 40)
    tread_color = (180, 180, 180)
    
    num_grooves = 4
    groove_positions = [45, 90, 135, 180]
    
    if wear_type == "new":
        width_range = 10
        blur_k = 1
        noise_level = 5
    elif wear_type == "serviceable":
        width_range = 5
        blur_k = 5
        noise_level = 15
    else:  # unusable / bald
        width_range = 0
        blur_k = 15
        noise_level = 25

    # Draw vertical stripes
    for x in range(224):
        # Determine groove depth based on position and optional asymmetry
        base_groove = False
        for pos in groove_positions:
            if abs(x - pos) < width_range:
                base_groove = True
                break
        
        # Apply wear asymmetry
        factor = 1.0
        if asymmetry_side == "left" and x < 90:
            factor = 0.2  # Diminish groove depth/contrast on left
        elif asymmetry_side == "right" and x > 134:
            factor = 0.2  # Diminish groove depth/contrast on right
            
        if base_groove and np.random.rand() < factor:
            img[:, x] = groove_color
        else:
            # Random tire texture noise
            val = int(180 + np.random.randint(-noise_level, noise_level + 1))
            val = max(0, min(255, val))
            img[:, x] = (val, val, val)
            
    # Apply Gaussian blur to simulate lens/tread wear smoothness
    if blur_k > 1:
        img = cv2.GaussianBlur(img, (blur_k, blur_k), 0)
        
    # Draw some horizontal tread sipes
    for y in range(10, 220, 20):
        # Only draw if not bald
        if wear_type != "unusable":
            for x in range(10, 210, 15):
                factor = 1.0
                if asymmetry_side == "left" and x < 90:
                    factor = 0.1
                elif asymmetry_side == "right" and x > 134:
                    factor = 0.1
                if np.random.rand() < (0.7 * factor):
                    cv2.line(img, (x, y), (x + 8, y + 2), (50, 50, 50), 1)

    # Save to file
    cv2.imwrite(filepath, img)

def generate_mock_tabular_data(filepath, n_rows=2000):
    """
    Generate synthetic automobile tire RUL tabular data matching Kaggle schema.
    """
    np.random.seed(42)
    
    # Categorical options
    vehicle_models = ["Sedan", "SUV", "Hatchback", "Truck", "Coupe"]
    fuel_types = ["Petrol", "Diesel", "Electric", "Hybrid"]
    transmissions = ["Automatic", "Manual"]
    countries = ["Germany", "USA", "India", "Japan", "UK"]
    tyre_brands = ["Michelin", "Bridgestone", "Continental", "Goodyear", "Pirelli"]
    tyre_sizes = ["205/55R16", "225/65R17", "245/40R18", "195/65R15"]
    tread_materials = ["Carbon Black", "Silica Compound", "Dual Compound"]
    tread_patterns = ["Symmetric", "Asymmetric", "Directional"]
    road_conditions = ["Smooth", "Rough", "Off-road"]
    weather_conditions = ["Humid", "Cold", "Dry", "Rainy"]
    retreaded_opts = ["No", "Yes"]
    axle_types = ["driven", "dead"]
    
    # Generate columns
    data = {
        "vehicle_model": np.random.choice(vehicle_models, n_rows),
        "fuel_type": np.random.choice(fuel_types, n_rows),
        "transmission_type": np.random.choice(transmissions, n_rows),
        "country": np.random.choice(countries, n_rows),
        "maximum_power(hp)": np.random.randint(80, 450, n_rows),
        "maximum_torque(N/m)": np.random.randint(120, 600, n_rows),
        "maximum_speed(km/h)": np.random.randint(140, 280, n_rows),
        "vehicle_acceleration(0-100 km/h in seconds)": np.round(np.random.uniform(4.0, 14.0, n_rows), 2),
        "vehicle_mileage(mpg)": np.round(np.random.uniform(15.0, 55.0, n_rows), 2),
        "vehicle_sprung_mass(kg)": np.random.randint(1000, 2500, n_rows),
        "steering_radius(m)": np.round(np.random.uniform(4.5, 6.5, n_rows), 2),
        "axle_type(driven/dead)": np.random.choice(axle_types, n_rows),
        "tyre_brand": np.random.choice(tyre_brands, n_rows),
        "tyre_size": np.random.choice(tyre_sizes, n_rows),
        "tread_material": np.random.choice(tread_materials, n_rows),
        "tread_pattern": np.random.choice(tread_patterns, n_rows),
        "tyre_camber_angle(degree)": np.round(np.random.uniform(-3.0, 3.0, n_rows), 2),
        "standard_tread_depth(mm)": np.random.choice([8.0, 7.5, 8.5], n_rows),
        "retreaded": np.random.choice(retreaded_opts, n_rows, p=[0.9, 0.1]),
        "road_condition": np.random.choice(road_conditions, n_rows),
        "weather_condition": np.random.choice(weather_conditions, n_rows),
    }
    
    # Expected tire life based on brand and standard depth
    expected_life = np.random.uniform(40000, 70000, n_rows)
    data["expected_tyre_life(km)"] = np.round(expected_life, 1)
    
    # Kilometers driven (less than or equal to expected life)
    km_driven = expected_life * np.random.uniform(0.0, 1.0, n_rows)
    data["kilometers_driven(km)"] = np.round(km_driven, 1)
    
    # Calculate RUL
    remaining_life = expected_life - km_driven
    
    # Modify remaining life based on camber angle, road condition, and weather
    # E.g. off-road and bad camber reduces RUL
    wear_factor = 1.0
    for i in range(n_rows):
        if data["road_condition"][i] == "Rough":
            wear_factor = 0.95
        elif data["road_condition"][i] == "Off-road":
            wear_factor = 0.85
            
        camber = abs(data["tyre_camber_angle(degree)"][i])
        if camber > 1.5:
            wear_factor *= 0.90
            
    remaining_life_modified = remaining_life * wear_factor
    data["remaining_useful_life(km)"] = np.round(np.clip(remaining_life_modified, 0, None), 1)
    
    # Current tread depth matches wear
    # Standard tire: new ~ 8.0mm, legal limit ~ 1.6mm.
    # Tread depth falls linearly with driven kilometers
    standard_depth = data["standard_tread_depth(mm)"]
    wear_ratio = data["kilometers_driven(km)"] / data["expected_tyre_life(km)"]
    current_depth = standard_depth * (1.0 - wear_ratio * 0.8) # Keep at least 20% tread at end of expected life
    # Add small noise
    current_depth += np.random.normal(0, 0.1, n_rows)
    # Clip between 1.0 and standard depth
    current_depth = np.clip(current_depth, 1.0, standard_depth)
    data["current_tread_depth(mm)"] = np.round(current_depth, 2)
    
    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)

def check_and_generate_mock_data(config):
    """
    Check if raw datasets exist. If not, generate synthetic mocks for testing.
    """
    raw_dir = config["data"]["raw_dir"]
    
    # 1. Tyre Condition Classification (Primary Image Set: New, Serviceable, Unusable)
    tyre_cond_dir = os.path.join(raw_dir, config["data"]["tyre_condition"])
    classes = ["New", "Serviceable", "Unusable"]
    has_tyre_cond = any(glob.glob(os.path.join(tyre_cond_dir, "**", "*.jpg"), recursive=True))
    
    if not has_tyre_cond:
        print("Dataset 'Tyre Condition Classification' not found. Generating mock images...")
        for c in classes:
            c_dir = os.path.join(tyre_cond_dir, c)
            os.makedirs(c_dir, exist_ok=True)
            # Generate 20 regular images per class
            for i in range(20):
                img_path = os.path.join(c_dir, f"mock_{c.lower()}_{i}.jpg")
                # Generate a few asymmetrical ones for testing alignment
                asym = None
                if c == "Serviceable" and i in [5, 6, 7]:
                    asym = "left"
                elif c == "Serviceable" and i in [8, 9, 10]:
                    asym = "right"
                generate_mock_image(img_path, wear_type=c.lower(), asymmetry_side=asym)
                
    # 2. TyreNet
    tyrenet_dir = os.path.join(raw_dir, config["data"]["tyrenet"])
    has_tyrenet = any(glob.glob(os.path.join(tyrenet_dir, "**", "*.jpg"), recursive=True))
    if not has_tyrenet:
        print("Dataset 'TyreNet' not found. Generating mock images...")
        for c in ["defective", "good"]:
            c_dir = os.path.join(tyrenet_dir, c)
            os.makedirs(c_dir, exist_ok=True)
            for i in range(10):
                img_path = os.path.join(c_dir, f"mock_tyrenet_{c}_{i}.jpg")
                generate_mock_image(img_path, wear_type="unusable" if c == "defective" else "new")

    # 3. Tyre Quality Classification
    tyre_qual_dir = os.path.join(raw_dir, config["data"]["tyre_quality"])
    has_tyre_qual = any(glob.glob(os.path.join(tyre_qual_dir, "**", "*.jpg"), recursive=True))
    if not has_tyre_qual:
        print("Dataset 'Tyre Quality' not found. Generating mock images...")
        for c in ["defective", "good"]:
            c_dir = os.path.join(tyre_qual_dir, c)
            os.makedirs(c_dir, exist_ok=True)
            for i in range(10):
                img_path = os.path.join(c_dir, f"mock_tyrequal_{c}_{i}.jpg")
                generate_mock_image(img_path, wear_type="unusable" if c == "defective" else "new")

    # 4. Tire Texture
    tire_tex_dir = os.path.join(raw_dir, config["data"]["tire_texture"])
    has_tire_tex = any(glob.glob(os.path.join(tire_tex_dir, "**", "*.jpg"), recursive=True))
    if not has_tire_tex:
        print("Dataset 'Tire Texture' not found. Generating mock images...")
        for c in ["cracked", "normal"]:
            c_dir = os.path.join(tire_tex_dir, c)
            os.makedirs(c_dir, exist_ok=True)
            for i in range(10):
                img_path = os.path.join(c_dir, f"mock_tiretex_{c}_{i}.jpg")
                generate_mock_image(img_path, wear_type="unusable" if c == "cracked" else "new")

    # 5. Synthetic Automobile-Tyre RUL Data (Tabular CSV)
    rul_dir = os.path.join(raw_dir, config["data"]["synthetic_rul"])
    csv_path = os.path.join(rul_dir, "synthetic_automobile_tyre_rul_data.csv")
    if not os.path.exists(csv_path):
        print("Dataset 'Synthetic Automobile-Tyre RUL Data' CSV not found. Generating mock CSV...")
        generate_mock_tabular_data(csv_path)

def load_tyre_condition_images(config):
    """
    Search recursively and return all paths to Tyre Condition images and their 3-class label mapping.
    Classes: {'New': 0, 'Serviceable': 1, 'Unusable': 2}
    """
    raw_dir = config["data"]["raw_dir"]
    tyre_cond_dir = os.path.join(raw_dir, config["data"]["tyre_condition"])
    
    classes_map = {"New": 0, "Serviceable": 1, "Unusable": 2}
    image_paths = []
    labels = []
    
    for c_name, c_id in classes_map.items():
        # Search for images under subdirectories containing the class name
        pattern1 = os.path.join(tyre_cond_dir, "**", c_name, "*.jpg")
        pattern2 = os.path.join(tyre_cond_dir, "**", c_name.lower(), "*.jpg")
        pattern3 = os.path.join(tyre_cond_dir, c_name, "*.jpg")
        
        files = glob.glob(pattern1, recursive=True) + glob.glob(pattern2, recursive=True) + glob.glob(pattern3, recursive=True)
        # Deduplicate
        files = list(set(files))
        
        for f in files:
            image_paths.append(f)
            labels.append(c_id)
            
    # If recursive search yielded nothing, try finding any folder structures
    if not image_paths:
        for root, dirs, files in os.walk(tyre_cond_dir):
            for file in files:
                if file.lower().endswith(".jpg"):
                    # Guess label from parent directory path
                    parent = os.path.basename(root)
                    for c_name, c_id in classes_map.items():
                        if c_name.lower() in parent.lower():
                            image_paths.append(os.path.join(root, file))
                            labels.append(c_id)
                            break
                            
    return image_paths, labels

def load_rul_df(config):
    """
    Load the RUL tabular CSV file.
    """
    raw_dir = config["data"]["raw_dir"]
    rul_dir = os.path.join(raw_dir, config["data"]["synthetic_rul"])
    csv_files = glob.glob(os.path.join(rul_dir, "*.csv"))
    if not csv_files:
        # Check standard name if none found via glob
        csv_path = os.path.join(rul_dir, "synthetic_automobile_tyre_rul_data.csv")
        if os.path.exists(csv_path):
            csv_files = [csv_path]
        else:
            raise FileNotFoundError(f"No CSV dataset found inside {rul_dir}")
            
    # Load first matching CSV
    df = pd.read_csv(csv_files[0])
    return df
