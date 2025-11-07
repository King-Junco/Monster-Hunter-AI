from torch.utils.data import Dataset
from PIL import Image

class MonsterHunterDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, self.labels[idx]
    
    def __len__(self):
        return len(self.image_paths)