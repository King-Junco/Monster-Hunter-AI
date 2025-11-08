from torchvision import transforms

# modify images so that i can get more use out of small dataset

train_transform = transforms.Compose([
    transforms.RandomRotation(30), # defualt 30
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.1, 0.1), # default (0.1, 0.1)
        scale=(0.9, 1.1), # Default (0.9, 1.1)
        shear=15 # default 0 (Added after run 6)
    ),
    transforms.ColorJitter(
        brightness=0.2, # default 0.2
        contrast=0.2,
        saturation=0.2,
        hue=0.1,
    ),

    # Added after Run 6
    transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
    transforms.GaussianBlur(kernel_size= 3, p =0.3), # Blur images with 30% probability
    transforms.RandomErasing(p=0.2), # Randomly erase parts of image with 0% probability
    # End of additions

    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
])