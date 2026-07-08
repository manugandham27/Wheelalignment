import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

class IntegratedGradients:
    """
    Integrated Gradients (IG) implementation for high-resolution,
    pixel-level wear attribution mapping. Satisfies completeness and linearity axioms.
    """
    def __init__(self, model):
        self.model = model

    def generate_attribution(self, input_tensor, target_class=None, steps=20):
        """
        Generate Integrated Gradients attribution map.
        input_tensor: shape (1, 3, 224, 224)
        """
        self.model.eval()
        device = input_tensor.device
        
        # Define baseline (black image)
        baseline = torch.zeros_like(input_tensor).to(device)
        
        # 1. Generate scaled inputs (linear interpolation path)
        scaled_inputs = []
        for i in range(steps + 1):
            alpha = float(i) / steps
            interpolated = baseline + alpha * (input_tensor - baseline)
            scaled_inputs.append(interpolated)
            
        # Concatenate into a batch
        scaled_inputs = torch.cat(scaled_inputs, dim=0) # (steps+1, 3, 224, 224)
        scaled_inputs.requires_grad_()
        
        # 2. Forward pass w.r.t visual branch only
        # Note: We pass zero IMU and tabular tensors since it is a batch of scaled inputs
        batch_size = scaled_inputs.size(0)
        dummy_imu = torch.zeros(batch_size, 100, 6, device=device)
        dummy_tab = torch.zeros(batch_size, 23, device=device)
        
        class_logits, _ = self.model(scaled_inputs, imu_seq=dummy_imu, tab_feats=dummy_tab)
        
        if target_class is None:
            # Predict from first index (original input, alpha=1.0)
            target_class = torch.argmax(class_logits[-1]).item()
            
        # Zero gradients
        self.model.zero_grad()
        
        # Extract target class logits
        target_logits = class_logits[:, target_class]
        
        # Backward pass to get gradients
        grads = torch.autograd.grad(outputs=target_logits, inputs=scaled_inputs,
                                    grad_outputs=torch.ones_like(target_logits),
                                    create_graph=False, retain_graph=False)[0]
                                    
        # 3. Compute Riemann sum approximation of the integral
        grads = grads.cpu().data.numpy()
        avg_grads = np.mean(grads, axis=0) # (3, 224, 224)
        
        # Multiplied by input - baseline
        delta = (input_tensor - baseline).cpu().squeeze(0).data.numpy() # (3, 224, 224)
        attribution = delta * avg_grads # (3, 224, 224)
        
        # Sum attribution across color channels
        attribution = np.sum(attribution, axis=0) # (224, 224)
        
        # Apply ReLU (focus on positive attributions)
        attribution = np.maximum(attribution, 0)
        
        # Normalize
        attr_min, attr_max = np.min(attribution), np.max(attribution)
        if attr_max > attr_min:
            attribution = (attribution - attr_min) / (attr_max - attr_min)
        else:
            attribution = np.zeros_like(attribution)
            
        return attribution

def apply_saliency_overlay(img_path, attribution, save_path):
    """
    Overlay attribution saliency onto original image using Jet colormap.
    """
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image at {img_path}")
        
    h, w, _ = img.shape
    
    # Resize attribution map to match original image size
    attr_resized = cv2.resize(attribution, (w, h))
    
    # 8-bit heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * attr_resized), cv2.COLORMAP_JET)
    
    # Overlay blend (50% original, 50% attribution map)
    overlay = cv2.addWeighted(img, 0.5, heatmap, 0.5, 0)
    
    cv2.imwrite(save_path, overlay)
    return save_path

def generate_ig_visualization(model, img_path, target_class, save_path, config):
    """
    Unified caller generating and saving Integrated Gradients wear attribution heatmap.
    """
    ig = IntegratedGradients(model)
    
    img_pil = Image.open(img_path).convert("RGB")
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = preprocess(img_pil).unsqueeze(0)  # Add batch
    
    model.cpu()
    input_tensor = input_tensor.cpu()
    
    attribution = ig.generate_attribution(input_tensor, target_class=target_class, steps=20)
    apply_saliency_overlay(img_path, attribution, save_path)
    return save_path
