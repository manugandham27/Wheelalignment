import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.models as models
from sklearn.model_selection import train_test_split

from utils.helper import load_config, get_device
from data.data_loader import check_and_generate_mock_data, load_tyre_condition_images
from models.wear_classifier.dataset import TireWearDataset
from models.wear_classifier.model import DualHeadTireCNN

class StudentMobileNet(nn.Module):
    """
    Lightweight Student Model using MobileNetV3-Small backbone.
    Trained via knowledge distillation to mimic the heavy multimodal teacher model.
    """
    def __init__(self, num_classes=3):
        super(StudentMobileNet, self).__init__()
        try:
            # Modern torchvision API
            weights = models.MobileNet_V3_Small_Weights.DEFAULT
            self.backbone = models.mobilenet_v3_small(weights=weights)
        except AttributeError:
            self.backbone = models.mobilenet_v3_small(pretrained=True)
            
        in_features = self.backbone.classifier[0].in_features
        # Replace classifier with identity
        self.backbone.classifier = nn.Identity()
        
        # Student dual heads
        self.fc_classifier = nn.Linear(in_features, num_classes)
        self.fc_regressor = nn.Linear(in_features, 1)

    def forward(self, img):
        # Student is image-only for extreme edge efficiency
        features = self.backbone(img)
        class_logits = self.fc_classifier(features)
        reg_output = self.fc_regressor(features).squeeze(-1)
        return class_logits, reg_output

def distillation_loss(student_logits, teacher_logits, labels, temperature=3.0, alpha=0.5):
    """
    Soft-target distillation loss using KL Divergence + CrossEntropy.
    """
    # Soft target loss
    soft_teacher = F.softmax(teacher_logits / temperature, dim=1)
    soft_student = F.log_softmax(student_logits / temperature, dim=1)
    loss_soft = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (temperature ** 2)
    
    # Hard target loss
    loss_hard = F.cross_entropy(student_logits, labels)
    
    return alpha * loss_soft + (1.0 - alpha) * loss_hard

def run_distillation():
    """
    Distillation training loop: transfers knowledge from the multimodal teacher to MobileNetV3.
    """
    config = load_config()
    device = get_device(config)
    
    # Load dataset
    image_paths, labels = load_tyre_condition_images(config)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=config["general"]["seed"], stratify=labels
    )
    
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    weak_label_cfg = config["models"]["wear_classifier"]["tread_depth_weak_labels"]
    train_dataset = TireWearDataset(train_paths, train_labels, transform=train_transforms, weak_label_cfg=weak_label_cfg)
    val_dataset = TireWearDataset(val_paths, val_labels, transform=val_transforms, weak_label_cfg=weak_label_cfg)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # 1. Load trained Teacher model
    teacher_path = config["models"]["wear_classifier"]["save_path"]
    backbone = config["models"]["wear_classifier"].get("backbone", "resnet18")
    teacher = DualHeadTireCNN(backbone_name=backbone, pretrained=False, num_classes=3)
    if os.path.exists(teacher_path):
        teacher.load_state_dict(torch.load(teacher_path, map_location=device))
        print("Loaded teacher model successfully.")
    else:
        print("Error: Distillation requires trained teacher checkpoint. Run train.py first.")
        return
        
    teacher.to(device)
    teacher.eval()
    
    # 2. Instantiate Student model
    student = StudentMobileNet(num_classes=3).to(device)
    optimizer = optim.Adam(student.parameters(), lr=0.001)
    criterion_reg = nn.MSELoss()
    
    epochs = 3
    print("Starting distillation training loop (mimicking teacher)...")
    for epoch in range(epochs):
        student.train()
        running_loss = 0.0
        
        for imgs, cls_lbls, reg_depths, imu_seqs, tab_feats in train_loader:
            imgs = imgs.to(device)
            cls_lbls = cls_lbls.to(device)
            reg_depths = reg_depths.to(device)
            imu_seqs = imu_seqs.to(device)
            tab_feats = tab_feats.to(device)
            
            # Forward pass teacher (no gradients needed)
            with torch.no_grad():
                teach_cls, teach_reg = teacher(imgs, imu_seq=imu_seqs, tab_feats=tab_feats)
                
            optimizer.zero_grad()
            
            # Forward pass student
            stud_cls, stud_reg = student(imgs)
            
            # Distillation losses
            loss_cls = distillation_loss(stud_cls, teach_cls, cls_lbls, temperature=3.0, alpha=0.6)
            
            # Regression mimicry: match ground truth and mimic teacher
            loss_reg = 0.5 * criterion_reg(stud_reg, reg_depths) + 0.5 * criterion_reg(stud_reg, teach_reg)
            
            loss = loss_cls + 0.5 * loss_reg
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * imgs.size(0)
            
        epoch_loss = running_loss / len(train_dataset)
        
        # Validation checks
        student.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for imgs, cls_lbls, _, _, _ in val_loader:
                imgs = imgs.to(device)
                cls_lbls = cls_lbls.to(device)
                
                stud_cls, _ = student(imgs)
                _, predicted = torch.max(stud_cls, 1)
                val_total += cls_lbls.size(0)
                val_correct += (predicted == cls_lbls).sum().item()
                
        val_acc = val_correct / val_total * 100
        print(f"Epoch {epoch+1:02d}/{epochs:02d} | Distill Loss: {epoch_loss:.4f} | Student Val Acc: {val_acc:.2f}%")
        
    student_save_path = os.path.join(os.path.dirname(teacher_path), "distilled_student.pth")
    torch.save(student.state_dict(), student_save_path)
    print(f"Distilled MobileNetV3 Student model saved successfully to: {student_save_path}")

if __name__ == "__main__":
    run_distillation()
