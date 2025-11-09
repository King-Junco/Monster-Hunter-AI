'''
predict.py - load saved model and make a prediction on a new image
'''
#imports 
import torch 
import torch.nn as nn
from PIL import Image
from pathlib import Path
import sys
import os
from torchvision import transforms

#import from code 
from main import MonsterHunterCNN, get_device, load_model

# Class names 
class_names = [
    'Amphibian', 'Bird Wyverns', 'Brute Wyverns', 'Carapaceons',
    'Elder Dragons', 'Fanged Beasts', 'Fanged Wyverns', 'Flying Wyverns',
    'Leviathans', 'Lynian', 'Neopterons', 'Piscine Wyverns', 'Temnocerans'
]

# Create prediction transform that auto-resizes to 256x256
predict_transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Always resize to match training
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
])

def predict_image(model, image_path, device):
    '''
    predict the class of a single image    
    '''
    try:
        # load and Transform image
        image = Image.open(image_path).convert('RGB')
        image_tensor = predict_transform(image).unsqueeze(0)
        image_tensor = image_tensor.to(device)
        
        # Make prediction
        model.eval()
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        predicted_class = class_names[predicted.item()]
        confidence_score = confidence.item() * 100 
        
        print(f'\n📸 Image: {os.path.basename(image_path)}')
        print(f'🎯 Predicted: {predicted_class}')
        print(f'📊 Confidence: {confidence_score:.2f}%')
        
        # Show top 3 predictions 
        print('🏆 Top 3 Predictions:')
        top3_prob, top3_idx = torch.topk(probabilities, 3)
        for i in range(3):
            class_name = class_names[top3_idx[0][i].item()]
            prob = top3_prob[0][i].item() * 100
            print(f'  {i+1}. {class_name}: {prob:.2f}%')
        
        return predicted_class, confidence_score
        
    except Exception as e:
        print(f'❌ Error processing {image_path}: {e}')
        return None, None

def predict_all_in_folder(model, folder_path, device):
    '''
    Predict classes for all images in a folder (using folder data/test)
    '''
    folder = Path(folder_path)
    if not folder.exists():
        print(f'❌ Folder Not Found: {folder_path}')
        return
    
    # Get All Images 
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp']
    image_files = [f for f in folder.iterdir()
                   if f.suffix.lower() in image_extensions]
    
    if not image_files:
        print(f'❌ No images found in folder: {folder_path}')
        return
    
    print(f'\n🔍 Found {len(image_files)} images to predict')
    print('=' * 60)
    
    for image_path in image_files:
        predict_image(model, str(image_path), device)
        print('-' * 60)

if __name__ == "__main__":
    # Get device
    device = get_device()
    
    # Load the Trained Model
    print('📂 Loading Model...')
    model = MonsterHunterCNN(num_classes=13)
    model = load_model(model, path='best_model_45.80.pth', device=device)
    print('✅ Model Loaded Successfully!\n')
    
    # Check if image path or folder path was provided
    if len(sys.argv) > 1:
        path = sys.argv[1]
        
        # Check if it's a file or folder
        if os.path.isfile(path):
            # Single Image
            predict_image(model, path, device)
        elif os.path.isdir(path):
            # Folder of Images
            predict_all_in_folder(model, path, device)
        else:
            print(f'❌ Invalid path: {path}')
    else:
        # Default to test folder
        test_folder = 'data/test'
        print(f'📁 No Path Provided. Using the Default folder: {test_folder}')
        predict_all_in_folder(model, test_folder, device)

'''
Usage:
Option 1: Predict on all images in data/test/
    python predict.py

Option 2: Predict on specific folder
    python predict.py data/test

Option 3: Predict on single image
    python predict.py data/test/rathalos.png
'''