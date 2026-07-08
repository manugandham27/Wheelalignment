import torch
import torch.nn as nn
import torchvision.models as models
from models.wear_classifier.imu_model import TireIMUModel

class DualHeadTireCNN(nn.Module):
    """
    Upscaled Multimodal Fusion Network:
    1. Visual Encoder: ResNet18 or ConvNeXt-T backbone extracting image features.
    2. Inertial Sequence Encoder: 1D CNN + LSTM extracting IMU features.
    3. Tabular Projection: Linear mapping of vehicle operational sensor vectors.
    4. Cross-Attention Fusion: Multihead Attention where visual representations
       attend to combined IMU + Tabular context keys and values.
    5. Dual Heads: Wear classification (3 classes) and Tread Depth regression.
    """
    def __init__(self, backbone_name="resnet18", pretrained=True, num_classes=3, num_tabular=23):
        super(DualHeadTireCNN, self).__init__()
        
        self.backbone_name = backbone_name.lower()
        
        # 1. Visual Backbone setup
        if self.backbone_name == "resnet18":
            try:
                weights = models.ResNet18_Weights.DEFAULT if pretrained else None
                self.backbone = models.resnet18(weights=weights)
            except AttributeError:
                self.backbone = models.resnet18(pretrained=pretrained)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            
        elif self.backbone_name == "convnext_tiny":
            try:
                weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
                self.backbone = models.convnext_tiny(weights=weights)
            except AttributeError:
                self.backbone = models.convnext_tiny(pretrained=pretrained)
            in_features = self.backbone.classifier[2].in_features
            # Replace classifier with Identity
            self.backbone.classifier = nn.Identity()
            
        elif self.backbone_name == "efficientnet_b0":
            try:
                weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
                self.backbone = models.efficientnet_b0(weights=weights)
            except AttributeError:
                self.backbone = models.efficientnet_b0(pretrained=pretrained)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}.")

        # Project visual features to standard fusion dimension (256)
        self.visual_project = nn.Linear(in_features, 256)
        
        # 2. IMU sequence encoder (1D CNN + LSTM)
        self.imu_encoder = TireIMUModel() # Outputs 128
        
        # 3. Tabular metadata encoder
        self.tab_encoder = nn.Sequential(
            nn.Linear(num_tabular, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        # Project combined sensor features (128 IMU + 128 Tabular = 256) to standard fusion dim
        self.sensor_project = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU()
        )

        # 4. Cross-Attention Fusion layer
        # Image acts as query (dim 256), IMU+Tabular acts as key & value (dim 256)
        self.cross_attention = nn.MultiheadAttention(embed_dim=256, num_heads=4, batch_first=True)
        
        # Layer norms for stabilization
        self.norm_visual = nn.LayerNorm(256)
        self.norm_sensor = nn.LayerNorm(256)
        
        # 5. Dual Prediction Heads
        self.fc_classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
        self.fc_regressor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, img, imu_seq=None, tab_feats=None):
        """
        Forward pass with multimodal fusion.
        img: (batch_size, 3, 224, 224)
        imu_seq: (batch_size, 100, 6) (optional)
        tab_feats: (batch_size, 23) (optional)
        """
        batch_size = img.size(0)
        device = img.device
        
        # 1. Extract image features
        img_feats = self.backbone(img) # (batch_size, in_features)
        img_proj = self.visual_project(img_feats) # (batch_size, 256)
        img_proj = self.norm_visual(img_proj)
        
        # 2. Extract IMU features (fallback to zero if missing)
        if imu_seq is None:
            imu_seq = torch.zeros(batch_size, 100, 6, device=device)
        imu_feats = self.imu_encoder(imu_seq) # (batch_size, 128)
        
        # 3. Extract Tabular features (fallback to zero if missing)
        if tab_feats is None:
            tab_feats = torch.zeros(batch_size, 23, device=device)
        tab_proj = self.tab_encoder(tab_feats) # (batch_size, 128)
        
        # Concatenate sensor features
        sensors_concat = torch.cat([imu_feats, tab_proj], dim=1) # (batch_size, 256)
        sensors_proj = self.sensor_project(sensors_concat) # (batch_size, 256)
        sensors_proj = self.norm_sensor(sensors_proj)
        
        # 4. Cross-Attention
        # Reshape for multihead attention -> requires (batch_size, seq_len, embed_dim)
        # We treat visual as a sequence of length 1, and sensor as a sequence of length 1
        q = img_proj.unsqueeze(1) # (batch_size, 1, 256)
        kv = sensors_proj.unsqueeze(1) # (batch_size, 1, 256)
        
        # Query attends to key/value context
        attn_out, _ = self.cross_attention(query=q, key=kv, value=kv) # (batch_size, 1, 256)
        
        # Residual connection
        fused = (q + attn_out).squeeze(1) # (batch_size, 256)
        
        # 5. Predict
        class_logits = self.fc_classifier(fused)
        reg_output = self.fc_regressor(fused).squeeze(-1) # (batch_size,)
        
        return class_logits, reg_output
