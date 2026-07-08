import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

class GradCAM:
    """
    Grad-CAM (Gradient-weighted Class Activation Mapping) implementation
    for visualizing regions of the tire tread that influenced the classifier's decision.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.forward_hook = target_layer.register_forward_hook(self.save_activation)
        self.backward_hook = target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        # grad_output is a tuple; we want the gradient w.r.t the output of the target layer
        self.gradients = grad_output[0]

    def remove_hooks(self):
        self.forward_hook.remove()
        self.backward_hook.remove()

    def generate_cam(self, input_tensor, target_class=None):
        """
        Generate CAM heatmap for a given input tensor.
        """
        self.model.eval()
        
        # Forward pass
        class_logits, _ = self.model(input_tensor)
        
        if target_class is None:
            target_class = torch.argmax(class_logits, dim=1).item()
            
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass w.r.t target class
        target_score = class_logits[0, target_class]
        target_score.backward()
        
        # Get gradients and activations
        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]
        
        # Global average pooling of gradients to get weights
        weights = np.mean(gradients, axis=(1, 2))
        
        # Weighted sum of activations
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        # Apply ReLU (we only care about features that positively correlate with the class)
        cam = np.maximum(cam, 0)
        
        # Resize to input dimensions (224x224)
        cam = cv2.resize(cam, (224, 224))
        
        # Normalize
        cam_min, cam_max = np.min(cam), np.max(cam)
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
            
        return cam

def apply_heatmap(img_path, cam, save_path):
    """
    Overlay 2D CAM heatmap onto the original image using OpenCV JET colormap.
    """
    # Load original image in color
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image at {img_path}")
        
    h, w, _ = img.shape
    
    # Resize CAM to match original image size
    cam_resized = cv2.resize(cam, (w, h))
    
    # Convert to 8-bit heatmap image
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    
    # Overlay heatmap on original image (blend with 60% original, 40% heatmap)
    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    
    cv2.imwrite(save_path, overlay)
    return save_path

def generate_gradcam_visualization(model, img_path, target_class, save_path, config):
    """
    Hook appropriate target layer, compute Grad-CAM, and save overlay heatmap image.
    """
    backbone_name = config["models"]["wear_classifier"]["backbone"].lower()
    
    # Locate target layer based on backbone architecture
    if backbone_name == "resnet18":
        # Final block of layer4 is a standard ResNet conv hook target
        target_layer = model.backbone.layer4[-1]
    elif backbone_name == "efficientnet_b0":
        # Final convolutional block in features extractor
        target_layer = model.backbone.features[-1]
    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")

    # Set up GradCAM hooks
    grad_cam = GradCAM(model, target_layer)
    
    # Load and transform image for model input
    img_pil = Image.open(img_path).convert("RGB")
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = preprocess(img_pil).unsqueeze(0)  # Add batch dimension
    
    # Ensure input tensor requires grad to propagate backward
    input_tensor.requires_grad_()
    
    # Put model on CPU for explainability generation to ensure compatibility
    model.cpu()
    input_tensor = input_tensor.cpu()
    
    try:
        cam = grad_cam.generate_cam(input_tensor, target_class=target_class)
        # Apply heatmap overlay
        apply_heatmap(img_path, cam, save_path)
    finally:
        # Crucial to clean up hooks to prevent memory leaks and issues on subsequent runs
        grad_cam.remove_hooks()
        
    return save_path
