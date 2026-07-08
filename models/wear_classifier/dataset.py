import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from data.data_loader import generate_simulated_imu

class TireWearDataset(Dataset):
    """
    Custom PyTorch Dataset for loading tire tread-wear images.
    Returns:
        - Image tensor (resized and normalized)
        - Classification label (0: New, 1: Serviceable, 2: Unusable)
        - Weakly labeled tread depth in mm (float regression target)
        - Simulated 100x6 time-series IMU sequence tensor (float32)
        - Simulated 23-dimensional tabular feature vector (float32)
    """
    def __init__(self, image_paths, labels, transform=None, weak_label_cfg=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
        # Load weak labels from configuration (default if none provided)
        if weak_label_cfg is None:
            weak_label_cfg = {"New": 8.0, "Serviceable": 4.0, "Unusable": 1.6}
            
        self.weak_depth_map = {
            0: weak_label_cfg.get("New", 8.0),
            1: weak_label_cfg.get("Serviceable", 4.0),
            2: weak_label_cfg.get("Unusable", 1.6)
        }

    def __len__(self):
        return len(self.image_paths)

    def _generate_simulated_tab_feats(self, label, idx):
        """
        Generate a simulated 23-dimensional tabular feature vector matching our RUL schema.
        Dimensions:
            0-10: projected numeric values (mileage, camber, standard depth, expected life, etc.)
            11-22: projected categorical one-hot values
        """
        np.random.seed(idx)
        
        # Base values depending on wear level
        if label == 0:  # New
            km_driven = np.random.uniform(0, 5000)
            expected_life = np.random.uniform(50000, 65000)
            camber = np.random.uniform(-0.5, 0.5)
            std_depth = 8.0
            cur_depth = std_depth - (km_driven / expected_life) * 6.4
        elif label == 1:  # Serviceable
            km_driven = np.random.uniform(15000, 35000)
            expected_life = np.random.uniform(50000, 65000)
            camber = np.random.uniform(-1.5, 1.5)
            std_depth = 8.0
            cur_depth = std_depth - (km_driven / expected_life) * 6.4
        else:  # Unusable
            km_driven = np.random.uniform(40000, 60000)
            expected_life = np.random.uniform(50000, 65000)
            camber = np.random.uniform(-2.5, 2.5)
            std_depth = 8.0
            cur_depth = std_depth - (km_driven / expected_life) * 6.4
            
        # Add random scaling to complete 23 numeric dimensions
        feats = [
            150.0 + np.random.normal(0, 20),      # max power
            220.0 + np.random.normal(0, 30),      # max torque
            200.0 + np.random.normal(0, 10),      # max speed
            8.5 + np.random.normal(0, 1),         # accel
            30.0 + np.random.normal(0, 3),        # mileage mpg
            1500.0 + np.random.normal(0, 100),    # sprung mass
            5.5 + np.random.normal(0, 0.5),       # steering radius
            camber,
            std_depth,
            expected_life,
            km_driven,
            cur_depth
        ]
        
        # Pad up to 23 features using one-hot indicators
        while len(feats) < 23:
            feats.append(np.random.choice([0.0, 1.0]))
            
        return np.array(feats, dtype=np.float32)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        img = Image.open(img_path).convert("RGB")
        
        # Transform
        if self.transform:
            img = self.transform(img)
            
        # Calculate weak continuous tread depth (mm) regression target
        base_depth = self.weak_depth_map[label]
        np.random.seed(idx)  # Ensure deterministic mapping for reproducibility
        noise = np.random.uniform(-0.4, 0.4)
        tread_depth = float(np.clip(base_depth + noise, 1.0, 9.0))
        
        # Generate simulated time-series IMU data (100 timesteps x 6 channels)
        imu_data = generate_simulated_imu(label, length=100, seed=idx)
        imu_tensor = torch.tensor(imu_data, dtype=torch.float32) # (100, 6)
        
        # Generate simulated tabular features
        tab_data = self._generate_simulated_tab_feats(label, idx)
        tab_tensor = torch.tensor(tab_data, dtype=torch.float32) # (23,)
        
        return img, int(label), torch.tensor(tread_depth, dtype=torch.float32), imu_tensor, tab_tensor
