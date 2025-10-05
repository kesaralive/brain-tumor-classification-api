import argparse
from utils.helpers import predict_directory, predict_single_image
from pathlib import Path



def main():
    parser = argparse.ArgumentParser(description='Brain Tumor Classification - Prediction')
    parser.add_argument('--model_path', type=str, required=True, 
                       help='Path to the trained model (.pth file)')
    parser.add_argument('--image_path', type=str, required=True,
                       help='Path to image or directory of images')
    parser.add_argument('--model_type', type=str, default='ResNet50',
                       choices=['CustomCNN', 'ResNet50', 'EfficientNet', 'ViT'],
                       help='Type of model architecture')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save output (image or CSV)')
    parser.add_argument('--class_names', nargs='+', 
                       default=['glioma', 'meningioma', 'no_tumor', 'pituitary'],
                       help='List of class names')
    parser.add_argument('--gradcam', action='store_true',
                       help='Show Grad-CAM visualization')
    parser.add_argument('--batch', action='store_true',
                       help='Process directory of images')
    
    args = parser.parse_args()
    
    # Check if path is directory
    if Path(args.image_path).is_dir():
        args.batch = True
    
    if args.batch:
        predict_directory(args)
    else:
        predict_single_image(args)
