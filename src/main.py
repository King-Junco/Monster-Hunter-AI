"""
main.py — portable PyTorch entry point for NVIDIA (CUDA) and AMD (DirectML)
"""
# Allow script to see data folder 
import sys, os 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data.augmentations.transforms import train_transform, val_transform
from data.dataset import MonsterHunterDataset
from sklearn.model_selection import train_test_split

# 🔍 Step 1: Detect the best available device
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

device = get_device()

# Example: placeholder model
class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)
    
    def forward(self, x):
        return self.fc(x)

model = DummyModel().to(device)

# 🔄 Step 2: Example of saving and loading cross-device
def save_model(model, path="model.pth"):
    torch.save(model.state_dict(), path)
    print(f"💾 Model saved to {path}")

def load_model(model, path="model.pth", device=device):
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    print(f"📂 Model loaded from {path} to {device}")
    return model

# Start CNN architecture training here
class MonsterHunterCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            # first conv layer
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 256 -> 128
            # second conv layer
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 128 -> 64
            # third conv layer
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 64 -> 32
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 32 * 32, 1024), # (256 channels, 32x32 feature map)
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()  # ADDED - calculates gradients
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = 100. * correct / total
        train_loss = running_loss / len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0.0 
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_acc = 100. * correct / total
        val_loss = val_loss / len(val_loader)
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.3f} | Val Acc: {val_acc:.2f}%')

        # Save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(model, 'best_model.pth')


# Example usage:
if __name__ == "__main__":
    # Hyperparameters
    num_classes = 15  # Number of Classes in dataset
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 50

    # Get device
    device = get_device()

    # Load dataset paths and labels

    # Directory
    data_dir = 'data/processed'

    # Class names 
    class_names = [
        'Amphibian', 'Bird Wyvern', 'Brute Wyvern', 'Carapaceons', 'Cephalopods',
        'Constructs', 'Elder Dragons', 'Fanged Beasts', 'Fanged Wyverns', 'Flying Wyverns',
        'Leviathans', 'Lynians', 'Neopterons', 'Piscine Wyverns', 'Temnocerans'
    ]

    # Collect all image paths and labels
    image_paths = []
    labels = []

    for class_idx, class_name in enumerate(class_names):
        class_folder = os.path.join(data_dir, f'({class_idx + 1}) {class_name}')

        # Get all images in folder 
        for img_name in os.listdir(class_folder):
            if img_name.endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(class_folder, img_name)
                image_paths.append(img_path)
                labels.append(class_idx)
        
    # Split Dataset into Traing and Validation (80-20 split)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels
        )

    print(f'Total images: {len(image_paths)}')
    print(f' Training images: {len(train_paths)}')
    print(f'Validation images: {len(val_paths)}')

    # Setup data loaders
    train_dataset = MonsterHunterDataset(
        image_paths = train_paths,
        labels = train_labels,
        transform = train_transform
    )

    # Validation dataset
    val_dataset = MonsterHunterDataset(
        image_paths = val_paths,
        labels = val_labels,
        transform = val_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model, criterion, and optimizer
    model = MonsterHunterCNN(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train_model(model, 
                train_loader, 
                val_loader, 
                criterion, 
                optimizer, 
                num_epochs, 
                device)
    # Save final model
    save_model(model, 'final_model.pth')