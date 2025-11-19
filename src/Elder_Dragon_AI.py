"""
elder_dragon_Ai.py - Binary classifier for Elder Dragon detection
Classifies monsters as: Elder Dragon (1) or Not Elder Dragon (0)
"""
import sys, os 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader
from data.augmentations.transforms import train_transform, val_transform
from data.dataset import MonsterHunterDataset
from sklearn.model_selection import train_test_split
from torchvision.models import ResNet18_Weights

# 🔍 Reuse the get_device from main.py
def get_device():
    try:
        import torch_directml
        device = torch_directml.device()
        print("✅ Using DirectML backend (AMD / Intel GPU detected)")
        return device
    except ImportError:
        if torch.cuda.is_available():
            print("✅ Using CUDA backend (NVIDIA GPU detected)")
            return torch.device("cuda")
        else:
            print("⚙️ Using CPU (no compatible GPU detected)")
            return torch.device("cpu")

# 🔄 Save/Load functions
def save_model(model, path="elder_dragon_model.pth"):
    torch.save(model.state_dict(), path)
    print(f"💾 Model saved to {path}")

def load_model(model, path="elder_dragon_model.pth", device=None):
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    print(f"📂 Model loaded from {path} to {device}")
    return model

# Binary Classifier Model - Elder Dragon or Not
class ElderDragonClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        
        # Freeze everything
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Binary classification head
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),  # Binary output (1 = Elder Dragon, 0 = Not)
            nn.Sigmoid()  # Output probability between 0 and 1
        )
    
    def forward(self, x):
        return self.model(x)

# Training function for binary classification
def train_binary_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.float().unsqueeze(1).to(device)  # Binary labels need to be float

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            predicted = (outputs > 0.5).float()  # Threshold at 0.5
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100. * correct / total
        train_loss = running_loss / len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.float().unsqueeze(1).to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                predicted = (outputs > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = 100. * correct / total
        val_loss = val_loss / len(val_loader)
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.3f} | Val Acc: {val_acc:.2f}%')

        # Save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(model, 'best_elder_dragon_model.pth')
            print(f"🎯 New best validation accuracy: {val_acc:.2f}%")

    print(f"\n✅ Training complete! Best validation accuracy: {best_val_acc:.2f}%")

# Main execution
if __name__ == "__main__":
    # Hyperparameters
    batch_size = 16
    learning_rate = 0.001
    num_epochs = 50

    # Get device (uses CPU for transfer learning compatibility)
    device = torch.device("cpu")
    print("⚙️ Using CPU for transfer learning")

    # Directory
    data_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'data',
        'processed'
    )

    print(f'Using data directory: {data_dir}')

    # All class names
    all_classes = [
        'Amphibian', 'Bird Wyverns', 'Brute Wyverns', 'Carapaceons',
        'Elder Dragons', 'Fanged Beasts', 'Fanged Wyverns', 'Flying Wyverns',
        'Leviathans', 'Lynian', 'Neopterons', 'Piscine Wyverns', 'Temnocerans'
    ]

    # Collect all image paths with binary labels
    image_paths = []
    labels = []

    for class_idx, class_name in enumerate(all_classes):
        class_folder = os.path.join(data_dir, f'({class_idx + 1}) {class_name}')
        
        # Determine if this is Elder Dragon class
        is_elder_dragon = 1 if class_name == 'Elder Dragons' else 0

        # Get all images in folder
        if os.path.exists(class_folder):
            for img_name in os.listdir(class_folder):
                if img_name.endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_folder, img_name)
                    image_paths.append(img_path)
                    labels.append(is_elder_dragon)  # Binary: 1 or 0

    # Count the distribution
    elder_count = sum(labels)
    not_elder_count = len(labels) - elder_count
    
    print(f'\n📊 Dataset Distribution:')
    print(f'Total images: {len(image_paths)}')
    print(f'Elder Dragons: {elder_count} ({elder_count/len(labels)*100:.1f}%)')
    print(f'Not Elder Dragons: {not_elder_count} ({not_elder_count/len(labels)*100:.1f}%)')

    # Split dataset (80-20 split)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels  # Ensures balanced split
    )

    print(f'\n📂 Split:')
    print(f'Training images: {len(train_paths)}')
    print(f'Validation images: {len(val_paths)}')

    # Create datasets
    train_dataset = MonsterHunterDataset(
        image_paths=train_paths,
        labels=train_labels,
        transform=train_transform
    )

    val_dataset = MonsterHunterDataset(
        image_paths=val_paths,
        labels=val_labels,
        transform=val_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    print("\n🔄 Initializing Elder Dragon Binary Classifier...")
    model = ElderDragonClassifier()
    model = model.to(device)
    print("✅ Model ready")

    # Binary Cross Entropy Loss for binary classification
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)

    print(f"\n🚀 Starting training for {num_epochs} epochs...")
    print("=" * 60)

    # Train the model
    train_binary_model(
        model, 
        train_loader, 
        val_loader, 
        criterion, 
        optimizer, 
        num_epochs, 
        device
    )

    # Save final model
    save_model(model, 'final_elder_dragon_model.pth')
    print("\n✅ Training complete! Models saved.")
    print("   - best_elder_dragon_model.pth (best validation accuracy)")
    print("   - final_elder_dragon_model.pth (final epoch)")