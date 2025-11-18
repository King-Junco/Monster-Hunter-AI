"""
evaluate_models.py - Generate confusion matrices and ROC curves
Works for both multi-class (main.py) and binary (elder_dragon_Ai.py) classifiers
"""
import sys, os 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader
from data.augmentations.transforms import val_transform
from data.dataset import MonsterHunterDataset
from sklearn.model_selection import train_test_split
import torchvision.models as models
from torchvision.models import ResNet18_Weights

# Import model architectures from your files
class MonsterHunterResNet18_Latest(nn.Module):
    """Multi-class classifier from main.py"""
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        
        for param in list(self.model.parameters())[:-10]:
            param.requires_grad = False
        
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3), 
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)

class ElderDragonClassifier(nn.Module):
    """Binary classifier from elder_dragon_Ai.py"""
    def __init__(self):
        super().__init__()
        self.model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        
        for param in self.model.parameters():
            param.requires_grad = False
        
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

def evaluate_multiclass_model(model, val_loader, class_names, device, save_prefix='multiclass'):
    """
    Evaluate multi-class model and generate confusion matrix + ROC curves
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    print("🔄 Evaluating multi-class model...")
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            # Get probabilities using softmax
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate accuracy
    accuracy = 100. * np.sum(all_preds == all_labels) / len(all_labels)
    print(f"✅ Validation Accuracy: {accuracy:.2f}%")
    
    # 1. CONFUSION MATRIX
    print("\n📊 Generating confusion matrix...")
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title(f'Confusion Matrix - Multi-Class Classification\nAccuracy: {accuracy:.2f}%', 
              fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_confusion_matrix.png', dpi=300, bbox_inches='tight')
    print(f"💾 Saved: {save_prefix}_confusion_matrix.png")
    plt.close()
    
    # 2. ROC CURVES (One-vs-Rest for multi-class)
    print("\n📈 Generating ROC curves...")
    
    # Binarize labels for multi-class ROC
    n_classes = len(class_names)
    y_bin = label_binarize(all_labels, classes=range(n_classes))
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], all_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_bin.ravel(), all_probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # Plot all ROC curves
    plt.figure(figsize=(12, 8))
    
    # Plot micro-average
    plt.plot(fpr["micro"], tpr["micro"],
             label=f'Micro-average (AUC = {roc_auc["micro"]:.3f})',
             color='deeppink', linestyle=':', linewidth=3)
    
    # Plot ROC curves for each class
    colors = plt.cm.Set3(np.linspace(0, 1, n_classes))
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                label=f'{class_names[i]} (AUC = {roc_auc[i]:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Multi-Class Classification (One-vs-Rest)', 
              fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_roc_curves.png', dpi=300, bbox_inches='tight')
    print(f"💾 Saved: {save_prefix}_roc_curves.png")
    plt.close()
    
    # 3. CLASSIFICATION REPORT
    print("\n📋 Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=3))
    
    # Save report to file
    with open(f'{save_prefix}_classification_report.txt', 'w') as f:
        f.write(f"Validation Accuracy: {accuracy:.2f}%\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(all_labels, all_preds, target_names=class_names, digits=3))
        f.write(f"\n\nROC AUC Scores (One-vs-Rest):\n")
        for i, class_name in enumerate(class_names):
            f.write(f"{class_name}: {roc_auc[i]:.3f}\n")
        f.write(f"Micro-average: {roc_auc['micro']:.3f}\n")
    
    print(f"💾 Saved: {save_prefix}_classification_report.txt")
    
    return accuracy, cm, roc_auc

def evaluate_binary_model(model, val_loader, device, save_prefix='binary'):
    """
    Evaluate binary model and generate confusion matrix + ROC curve
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    print("🔄 Evaluating binary model...")
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.float().to(device)
            
            outputs = model(inputs).squeeze()
            predicted = (outputs > 0.5).float()
            
            # Handle both single item and batch cases
            if predicted.dim() == 0:
                predicted = predicted.unsqueeze(0)
                outputs = outputs.unsqueeze(0)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(outputs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate accuracy
    accuracy = 100. * np.sum(all_preds == all_labels) / len(all_labels)
    print(f"✅ Validation Accuracy: {accuracy:.2f}%")
    
    # 1. CONFUSION MATRIX
    print("\n📊 Generating confusion matrix...")
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Not Elder Dragon', 'Elder Dragon'],
                yticklabels=['Not Elder Dragon', 'Elder Dragon'],
                cbar_kws={'label': 'Count'})
    plt.title(f'Confusion Matrix - Elder Dragon Binary Classification\nAccuracy: {accuracy:.2f}%', 
              fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_confusion_matrix.png', dpi=300, bbox_inches='tight')
    print(f"💾 Saved: {save_prefix}_confusion_matrix.png")
    plt.close()
    
    # 2. ROC CURVE
    print("\n📈 Generating ROC curve...")
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve - Elder Dragon Binary Classification', 
              fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_roc_curve.png', dpi=300, bbox_inches='tight')
    print(f"💾 Saved: {save_prefix}_roc_curve.png")
    plt.close()
    
    # 3. CLASSIFICATION REPORT
    print("\n📋 Classification Report:")
    report = classification_report(all_labels, all_preds, 
                                   target_names=['Not Elder Dragon', 'Elder Dragon'],
                                   digits=3)
    print(report)
    
    # Save report to file
    with open(f'{save_prefix}_classification_report.txt', 'w') as f:
        f.write(f"Validation Accuracy: {accuracy:.2f}%\n")
        f.write(f"ROC AUC Score: {roc_auc:.3f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write(f"\n\nConfusion Matrix:\n")
        f.write(f"True Negatives: {cm[0, 0]}\n")
        f.write(f"False Positives: {cm[0, 1]}\n")
        f.write(f"False Negatives: {cm[1, 0]}\n")
        f.write(f"True Positives: {cm[1, 1]}\n")
    
    print(f"💾 Saved: {save_prefix}_classification_report.txt")
    
    return accuracy, cm, roc_auc

def load_validation_data_multiclass():
    """Load validation data for multi-class model"""
    data_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'data',
        'processed'
    )
    
    class_names = [
        'Amphibian', 'Bird Wyverns', 'Brute Wyverns', 'Carapaceons',
        'Elder Dragons', 'Fanged Beasts', 'Fanged Wyverns', 'Flying Wyverns',
        'Leviathans', 'Lynian', 'Neopterons', 'Piscine Wyverns', 'Temnocerans'
    ]
    
    image_paths = []
    labels = []
    
    for class_idx, class_name in enumerate(class_names):
        class_folder = os.path.join(data_dir, f'({class_idx + 1}) {class_name}')
        for img_name in os.listdir(class_folder):
            if img_name.endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(class_folder, img_name)
                image_paths.append(img_path)
                labels.append(class_idx)
    
    # Use same split as training (random_state=42)
    _, val_paths, _, val_labels = train_test_split(
        image_paths, labels,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )
    
    val_dataset = MonsterHunterDataset(
        image_paths=val_paths,
        labels=val_labels,
        transform=val_transform
    )
    
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    return val_loader, class_names

def load_validation_data_binary():
    """Load validation data for binary model"""
    data_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'data',
        'processed'
    )
    
    all_classes = [
        'Amphibian', 'Bird Wyverns', 'Brute Wyverns', 'Carapaceons',
        'Elder Dragons', 'Fanged Beasts', 'Fanged Wyverns', 'Flying Wyverns',
        'Leviathans', 'Lynian', 'Neopterons', 'Piscine Wyverns', 'Temnocerans'
    ]
    
    image_paths = []
    labels = []
    
    for class_idx, class_name in enumerate(all_classes):
        class_folder = os.path.join(data_dir, f'({class_idx + 1}) {class_name}')
        is_elder_dragon = 1 if class_name == 'Elder Dragons' else 0
        
        if os.path.exists(class_folder):
            for img_name in os.listdir(class_folder):
                if img_name.endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_folder, img_name)
                    image_paths.append(img_path)
                    labels.append(is_elder_dragon)
    
    # Use same split as training (random_state=42)
    _, val_paths, _, val_labels = train_test_split(
        image_paths, labels,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )
    
    val_dataset = MonsterHunterDataset(
        image_paths=val_paths,
        labels=val_labels,
        transform=val_transform
    )
    
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    return val_loader

if __name__ == "__main__":
    device = torch.device("cpu")
    print("⚙️ Using CPU for evaluation\n")
    
    # Create reports directory structure
    base_report_dir = 'reports'
    os.makedirs(base_report_dir, exist_ok=True)
    
    # Ask user which model to evaluate
    print("Which model would you like to evaluate?")
    print("1. Multi-class model (main.py)")
    print("2. Binary Elder Dragon model (elder_dragon_Ai.py)")
    print("3. Both models")
    
    choice = input("\nEnter choice (1/2/3): ").strip()
    
    evaluated_models = []
    
    if choice in ['1', '3']:
        print("\n" + "="*60)
        print("EVALUATING MULTI-CLASS MODEL")
        print("="*60)
        
        # Load multi-class model
        model_path = input("Enter path to multi-class model (default: best_model.pth): ").strip()
        if not model_path:
            model_path = 'best_model.pth'
        
        if os.path.exists(model_path):
            # Extract model name from path (without extension)
            model_name = os.path.splitext(os.path.basename(model_path))[0]
            
            # Create subdirectory with model name
            model_dir = os.path.join(base_report_dir, model_name)
            os.makedirs(model_dir, exist_ok=True)
            
            model = MonsterHunterResNet18_Latest(num_classes=13)
            state_dict = torch.load(model_path, map_location=device, weights_only=False)
            model.load_state_dict(state_dict)
            model.to(device)
            print(f"✅ Loaded model from {model_path}")
            
            val_loader, class_names = load_validation_data_multiclass()
            save_prefix = os.path.join(model_dir, 'evaluation')
            evaluate_multiclass_model(model, val_loader, class_names, device, 
                                     save_prefix=save_prefix)
            evaluated_models.append((model_name, 'multiclass'))
        else:
            print(f"❌ Model file not found: {model_path}")
    
    if choice in ['2', '3']:
        print("\n" + "="*60)
        print("EVALUATING BINARY ELDER DRAGON MODEL")
        print("="*60)
        
        # Load binary model
        model_path = input("Enter path to binary model (default: best_elder_dragon_model.pth): ").strip()
        if not model_path:
            model_path = 'best_elder_dragon_model.pth'
        
        if os.path.exists(model_path):
            # Extract model name from path (without extension)
            model_name = os.path.splitext(os.path.basename(model_path))[0]
            
            # Create subdirectory with model name
            model_dir = os.path.join(base_report_dir, model_name)
            os.makedirs(model_dir, exist_ok=True)
            
            model = ElderDragonClassifier()
            state_dict = torch.load(model_path, map_location=device, weights_only=False)
            model.load_state_dict(state_dict)
            model.to(device)
            print(f"✅ Loaded model from {model_path}")
            
            val_loader = load_validation_data_binary()
            save_prefix = os.path.join(model_dir, 'evaluation')
            evaluate_binary_model(model, val_loader, device, 
                                save_prefix=save_prefix)
            evaluated_models.append((model_name, 'binary'))
        else:
            print(f"❌ Model file not found: {model_path}")
    
    print("\n" + "="*60)
    print("✅ EVALUATION COMPLETE!")
    print("="*60)
   