import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

from utils.helper import load_config, get_device, create_dirs_if_not_exist
from data.data_loader import check_and_generate_mock_data, load_tyre_condition_images
from models.wear_classifier.dataset import TireWearDataset
from models.wear_classifier.model import DualHeadTireCNN

def train_model():
    """
    Train the upscaled multimodal fusion wear classifier.
    Incorporates ConvNeXt-T, LSTM IMU encoder, and Cross-Attention context modeling.
    """
    config = load_config()
    create_dirs_if_not_exist(config)
    
    # Ensure datasets exist
    check_and_generate_mock_data(config)
    
    # Load images and labels
    image_paths, labels = load_tyre_condition_images(config)
    
    if len(image_paths) == 0:
        raise ValueError("No images found for training.")

    print(f"Loaded {len(image_paths)} images from Tyre Condition Classification Dataset.")

    # Stratified split into train and validation sets (80/20)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=config["general"]["seed"], stratify=labels
    )
    
    print(f"Train size: {len(train_paths)}, Val size: {len(val_paths)}")

    # Define transforms
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Dataset & Dataloaders
    weak_label_cfg = config["models"]["wear_classifier"]["tread_depth_weak_labels"]
    train_dataset = TireWearDataset(train_paths, train_labels, transform=train_transforms, weak_label_cfg=weak_label_cfg)
    val_dataset = TireWearDataset(val_paths, val_labels, transform=val_transforms, weak_label_cfg=weak_label_cfg)

    batch_size = config["models"]["wear_classifier"]["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Initialize model
    backbone = config["models"]["wear_classifier"].get("backbone", "resnet18")
    pretrained = config["models"]["wear_classifier"]["pretrained"]
    num_classes = config["models"]["wear_classifier"]["num_classes"]
    
    device = get_device(config)
    print(f"Training upscaled model on device: {device} (Backbone: {backbone})")
    
    model = DualHeadTireCNN(backbone_name=backbone, pretrained=pretrained, num_classes=num_classes)
    model.to(device)

    # Loss and optimizer
    criterion_cls = nn.CrossEntropyLoss()
    criterion_reg = nn.MSELoss()
    
    lr = config["models"]["wear_classifier"]["learning_rate"]
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    epochs = config["models"]["wear_classifier"]["epochs"]
    best_val_loss = float("inf")
    save_path = config["models"]["wear_classifier"]["save_path"]
    
    # Ensure directory containing save_path exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print("Starting training loop...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_cls_loss = 0.0
        running_reg_loss = 0.0
        correct = 0
        total = 0
        
        for imgs, cls_lbls, reg_depths, imu_seqs, tab_feats in train_loader:
            imgs = imgs.to(device)
            cls_lbls = cls_lbls.to(device)
            reg_depths = reg_depths.to(device)
            imu_seqs = imu_seqs.to(device)
            tab_feats = tab_feats.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass with multimodal inputs
            cls_out, reg_out = model(imgs, imu_seq=imu_seqs, tab_feats=tab_feats)
            
            # Losses
            loss_cls = criterion_cls(cls_out, cls_lbls)
            loss_reg = criterion_reg(reg_out, reg_depths)
            loss = loss_cls + 0.5 * loss_reg
            
            loss.backward()
            optimizer.step()
            
            # Stats
            running_loss += loss.item() * imgs.size(0)
            running_cls_loss += loss_cls.item() * imgs.size(0)
            running_reg_loss += loss_reg.item() * imgs.size(0)
            
            _, predicted = torch.max(cls_out, 1)
            total += cls_lbls.size(0)
            correct += (predicted == cls_lbls).sum().item()
            
        epoch_loss = running_loss / len(train_dataset)
        epoch_cls_loss = running_cls_loss / len(train_dataset)
        epoch_reg_loss = running_reg_loss / len(train_dataset)
        epoch_acc = correct / total * 100
        
        # Validation pass
        model.eval()
        val_loss = 0.0
        val_cls_loss = 0.0
        val_reg_loss = 0.0
        val_correct = 0
        val_total = 0
        val_mae = 0.0
        
        with torch.no_grad():
            for imgs, cls_lbls, reg_depths, imu_seqs, tab_feats in val_loader:
                imgs = imgs.to(device)
                cls_lbls = cls_lbls.to(device)
                reg_depths = reg_depths.to(device)
                imu_seqs = imu_seqs.to(device)
                tab_feats = tab_feats.to(device)
                
                cls_out, reg_out = model(imgs, imu_seq=imu_seqs, tab_feats=tab_feats)
                
                loss_cls = criterion_cls(cls_out, cls_lbls)
                loss_reg = criterion_reg(reg_out, reg_depths)
                loss = loss_cls + 0.5 * loss_reg
                
                val_loss += loss.item() * imgs.size(0)
                val_cls_loss += loss_cls.item() * imgs.size(0)
                val_reg_loss += loss_reg.item() * imgs.size(0)
                
                _, predicted = torch.max(cls_out, 1)
                val_total += cls_lbls.size(0)
                val_correct += (predicted == cls_lbls).sum().item()
                
                val_mae += torch.abs(reg_out - reg_depths).sum().item()
                
        epoch_val_loss = val_loss / len(val_dataset)
        epoch_val_cls_loss = val_cls_loss / len(val_dataset)
        epoch_val_reg_loss = val_reg_loss / len(val_dataset)
        epoch_val_acc = val_correct / val_total * 100
        epoch_val_mae = val_mae / len(val_dataset)
        
        print(f"Epoch {epoch+1:02d}/{epochs:02d} | "
              f"Train Loss: {epoch_loss:.4f} (Cls: {epoch_cls_loss:.4f}, Reg: {epoch_reg_loss:.4f}), Acc: {epoch_acc:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f} (Cls: {epoch_val_cls_loss:.4f}, Reg: {epoch_val_reg_loss:.4f}), Acc: {epoch_val_acc:.2f}%, MAE: {epoch_val_mae:.2f}mm")
        
        # Save best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), save_path)
            print(f"  --> Saved new best model checkpoint to {save_path}")

    print("Training completed successfully!")

if __name__ == "__main__":
    train_model()
