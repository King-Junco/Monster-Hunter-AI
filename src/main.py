"""
main.py — portable PyTorch entry point for NVIDIA (CUDA) and AMD (DirectML)
"""

import torch

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
import torch.nn as nn
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

# Example usage:
if __name__ == "__main__":
    # Save once
    save_model(model)

    # Load again (e.g., after moving to another machine)
    model = load_model(DummyModel(), device=device)
