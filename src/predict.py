'''
predict.py - load saved model and make a prediction on a new image
Works with: CNN from scratch, ResNet18 transfer learning, and Elder Dragon binary classifier
'''
#imports 
import torch 
import torch.nn as nn
from PIL import Image
from pathlib import Path
import sys
import os
from torchvision import transforms
import argparse

# Import model architectures
from main import MonsterHunterResNet18_Latest, get_device, load_model
from Elder_Dragon_AI import ElderDragonClassifier


# Class names for multi-class classification
class_names = [
    'Amphibian', 'Bird Wyverns', 'Brute Wyverns', 'Carapaceons',
    'Elder Dragons', 'Fanged Beasts', 'Fanged Wyverns', 'Flying Wyverns',
    'Leviathans', 'Lynian', 'Neopterons', 'Piscine Wyverns', 'Temnocerans'
]

# Binary class names for Elder Dragon classifier
binary_class_names = ['Not Elder Dragon', 'Elder Dragon']

# Create prediction transform that auto-resizes to 256x256
predict_transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Always resize to match training
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
])

def predict_image_multiclass(model, image_path, device):
    '''
    Predict the class of a single image (13-class classification)
    '''
    try:
        # Load and Transform image
        image = Image.open(image_path).convert('RGB')
        image_tensor = predict_transform(image).unsqueeze(0)
        
        # Use CPU for inference if DirectML
        inference_device = torch.device('cpu') if 'privateuseone' in str(device) else device
        image_tensor = image_tensor.to(inference_device)
        
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

def predict_image_binary(model, image_path, device):
    '''
    Predict Elder Dragon vs Not Elder Dragon (binary classification)
    '''
    try:
        # Load and Transform image
        image = Image.open(image_path).convert('RGB')
        image_tensor = predict_transform(image).unsqueeze(0)
        
        # Use CPU for inference if DirectML
        inference_device = torch.device('cpu') if 'privateuseone' in str(device) else device
        image_tensor = image_tensor.to(inference_device)
        
        # Make prediction
        model.eval()
        with torch.no_grad():
            output = model(image_tensor)
            probability = output.item()  # Single probability value
        
        predicted_class_idx = 1 if probability > 0.5 else 0
        predicted_class = binary_class_names[predicted_class_idx]
        confidence_score = probability * 100 if predicted_class_idx == 1 else (1 - probability) * 100
        
        print(f'\n📸 Image: {os.path.basename(image_path)}')
        print(f'🎯 Predicted: {predicted_class}')
        print(f'📊 Confidence: {confidence_score:.2f}%')
        print(f'🔢 Elder Dragon Probability: {probability * 100:.2f}%')
        
        return predicted_class, confidence_score
        
    except Exception as e:
        print(f'❌ Error processing {image_path}: {e}')
        return None, None

def predict_all_in_folder(model, folder_path, device, model_type='multiclass'):
    '''
    Predict classes for all images in a folder
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
        if model_type == 'binary':
            predict_image_binary(model, str(image_path), device)
        else:
            predict_image_multiclass(model, str(image_path), device)
        print('-' * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict Monster Hunter classes')
    parser.add_argument('path', nargs='?', default='data/test',
                        help='Path to image or folder (default: data/test)')
    parser.add_argument('--model', '-m', default='best_model.pth',
                        help='Path to model file (default: best_model.pth)')
    parser.add_argument('--type', '-t', choices=['multiclass', 'binary'], default='multiclass',
                        help='Model type: multiclass (13 classes) or binary (Elder Dragon)')
    
    args = parser.parse_args()
    
    # Smart model path resolution - check multiple locations
    model_path = args.model
    possible_paths = [
        model_path,  # Try exact path first
        os.path.join('..', model_path),  # Try parent directory (project root)
        os.path.join(os.path.dirname(__file__), model_path),  # Try script directory
        os.path.join(os.path.dirname(__file__), '..', model_path),  # Try parent of script dir
    ]
    
    # Find the first path that exists
    found_path = None
    for path in possible_paths:
        if os.path.exists(path):
            found_path = path
            break
    
    if found_path is None:
        print(f'❌ Error: Model file not found: {args.model}')
        print(f'\n🔍 Searched in these locations:')
        for path in possible_paths:
            print(f'   - {os.path.abspath(path)}')
        
        print(f'\n💡 Available .pth files found:')
        # Check current directory
        print(f'\n   In current directory ({os.getcwd()}):')
        local_models = [f for f in os.listdir('.') if f.endswith('.pth')]
        if local_models:
            for mf in sorted(local_models):
                print(f'      - {mf}')
        else:
            print('      (none)')
        
        # Check parent directory
        parent_dir = os.path.join('..', '')
        if os.path.exists(parent_dir):
            print(f'\n   In parent directory ({os.path.abspath(parent_dir)}):')
            parent_models = [f for f in os.listdir(parent_dir) if f.endswith('.pth')]
            if parent_models:
                for mf in sorted(parent_models):
                    print(f'      - {mf}')
            else:
                print('      (none)')
        
        print('\n📌 Tips:')
        print('   1. Specify full path: --model ../best_elder_dragon_model_86.82.pth')
        print('   2. Or move model files to src/ directory')
        print('   3. Or run from project root instead of src/')
        sys.exit(1)
    
    # Use the found path
    model_path = found_path
    print(f'📂 Loading Model: {os.path.basename(model_path)}')
    print(f'📍 From location: {os.path.abspath(model_path)}')
    print(f'🔍 Model Type: {args.type}\n')
    
    # Get device
    device = get_device()
    
    # Force CPU for DirectML compatibility when loading models
    load_device = device
    if 'privateuseone' in str(device):  # DirectML device
        print('⚠️  DirectML detected - using CPU for model loading (better compatibility)')
        load_device = torch.device('cpu')
    
    # Load the appropriate model
    try:
        if args.type == 'binary':
            # Load Elder Dragon binary classifier
            model = ElderDragonClassifier()
            model = load_model(model, path=model_path, device=load_device)
            print('✅ Elder Dragon Binary Classifier Loaded!\n')
        else:
            # Load multi-class classifier (ResNet18 or CNN)
            model = MonsterHunterResNet18_Latest(num_classes=13)
            model = load_model(model, path=model_path, device=load_device)
            print('✅ Multi-Class Model Loaded!\n')
    except Exception as e:
        print(f'❌ Error loading model: {e}')
        print('\n💡 Tips:')
        print('   - Make sure the model type matches the file (--type binary for elder dragon models)')
        print('   - Check if the model file is corrupted')
        sys.exit(1)
    
    # Check if image path or folder path was provided
    path = args.path
    
    # Smart path resolution for data folders too
    if not os.path.exists(path):
        possible_data_paths = [
            path,
            os.path.join('..', path),  # Try parent directory
            os.path.join(os.path.dirname(__file__), path),  # Try script directory
            os.path.join(os.path.dirname(__file__), '..', path),  # Try parent of script
        ]
        
        found_data_path = None
        for p in possible_data_paths:
            if os.path.exists(p):
                found_data_path = p
                break
        
        if found_data_path is None:
            print(f'❌ Invalid path: {path}')
            print(f'\n🔍 Searched in these locations:')
            for p in possible_data_paths:
                print(f'   - {os.path.abspath(p)}')
            print('\n💡 Make sure the path exists or use absolute path')
            sys.exit(1)
        
        path = found_data_path
    
    print(f'📁 Using path: {os.path.abspath(path)}\n')
    
    # Check if it's a file or folder
    if os.path.isfile(path):
        # Single Image
        if args.type == 'binary':
            predict_image_binary(model, path, load_device)
        else:
            predict_image_multiclass(model, path, load_device)
    elif os.path.isdir(path):
        # Folder of Images
        predict_all_in_folder(model, path, load_device, model_type=args.type)
    else:
        print(f'❌ Invalid path: {path}')

'''
Usage Examples:

1. Multi-class classification (13 classes):
   # Predict on all images in data/test/ using best_model.pth
   python predict.py

   # Predict on specific folder
   python predict.py data/test --model best_model.pth

   # Predict on single image
   python predict.py data/test/rathalos.png --model final_model.pth

2. Binary classification (Elder Dragon detection):
   # Predict on all images in data/test/
   python predict.py data/test --model best_elder_dragon_model.pth --type binary

   # Predict on single image
   python predict.py data/test/rathalos.png --model final_elder_dragon_model.pth --type binary
   
   # Short form
   python predict.py data/test -m best_elder_dragon_model.pth -t binary
'''