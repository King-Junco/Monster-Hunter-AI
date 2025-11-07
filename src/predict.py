'''
predict.py - load saved model and make a prediction on a new image
'''
#imports 
import torch 
import torch.nn as nn
from PIL import Image
import sys
import os

#import from code 
from main import MonsterHunterCNN, get_device, load_model
from data.augmentations.transforms import val_transform

# Class names 
class_names = [
        'Amphibian', 'Bird Wyvern', 'Brute Wyvern', 'Carapaceons', 'Cephalopods',
        'Constructs', 'Elder Dragons', 'Fanged Beasts', 'Fanged Wyverns', 'Flying Wyverns',
        'Leviathans', 'Lynians', 'Neopterons', 'Piscine Wyverns', 'Temnocerans'
        ]

def predict_image(model, image_path, device):
    '''
    predict the class of a single image    
    '''

    # load and Transform image
    image = Image.open(image_path).convert('RGB')
    image_tensor = val_transform(image).unsqueeze(0)
    image_tensor = image_tensor.to(device)

    # Make prediction
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    predicted_class = class_names[predicted.item()]
    confidence_score = confidence.item() * 100 

    print(f'\n Image: {os.path.basename(image_path)}')
    print(f'\n Predicted Class: {predicted_class}')
    print(f'\n Confidence Score: {confidence_score:.2f}%')

    # Show top 3 predictions 
    print('\n Top 3 Predictions:')
    top3_prob, top3_idx = torch.topk(probabilities, 3)
    for i in range(3):
        class_name = class_names[top3_idx[0][i].item()]
        prob = top3_prob[0][i].item() * 100
        print(f'  {i+1}. {class_name}: {prob:.2f}%')

    return predicted_class, confidence_score

def predict_all_in_folder(model, folder_path, device):
    '''
    Predict classes for all images in a folder (useing folder data/test)
    '''

    folder = path(folder_path)

    if not folder.exists():
        print(f'Folder Not Found: {folder_path}')
        return
    
    # Get All Images 
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp',]
    image_files = [f for f in folder.iterdir()
                   if f.suffix.lower() in image_extensions
                   ]
    
    if not image_files:
        print(f'No images found in folder: {folder_path}')
        return
    
    print(f'\n Found {len(image_files)} images to predict')
    print('=' * 60)

    for image_path in image_files:
        predict_image(model, str(image_path), device)
        print('=' * 60) 

if __name__ == "__main__":

    # Get device
    device = get_device()

    # Load the Trained Model
    print('Loading Model...')
    model = MonsterHunterCNN(num_classes=15)
    model = load_model(model, path='best_model.pth', device=device)
    print('Model Loaded Successfully!\n')

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
            print(f'Invalid path: {path}')
    else:
        # Default to test folder
        test_folder = 'data/test'
        print(f'No Path Provided. Using the Default folder: {test_folder}')
        predict_all_in_folder(model, test_folder, device)

'''
Option 1: Predict on all images in data/test/: python predict.py
Option 2: Predict on specific folder: python predict.py data/test
Option 3: Predict on single image: python predict.py data/test/rathalos.png
'''
