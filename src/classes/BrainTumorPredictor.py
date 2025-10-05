import torch
import cv2
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

from classes.CustomCNN import CustomCNN
from classes.ResNetClassifier import ResNetClassifier
from classes.EfficientNetClassifier import EfficientNetClassifier
from classes.VisionTransformer import VisionTransformer
from classes.GradCAM import GradCAM

class BrainTumorPredictor:
    """Main predictor class for brain tumor classification"""
    
    def __init__(self, model_path, model_type='ResNet50', class_names=None, device=None):
        """
        Initialize the predictor
        
        Args:
            model_path: Path to the .pth model file
            model_type: Type of model ('CustomCNN', 'ResNet50', 'EfficientNet', 'ViT')
            class_names: List of class names (default: ['glioma', 'meningioma', 'notumor', 'pituitary'])
            device: Device to run inference on (default: auto-detect)
        """
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = model_type
        
        # Default class names
        self.class_names = class_names if class_names else [
            'glioma', 'meningioma', 'no_tumor', 'pituitary'
        ]
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()
        
        print(f"✓ Model loaded successfully on {self.device}")
        print(f"✓ Model type: {model_type}")
        print(f"✓ Classes: {', '.join(self.class_names)}")
    
    def _load_model(self, model_path):
        """Load the trained model"""
        num_classes = len(self.class_names)
        
        # Initialize model architecture based on type
        if self.model_type == 'CustomCNN':
            model = CustomCNN(num_classes=num_classes)
        elif self.model_type == 'ResNet50':
            model = ResNetClassifier(num_classes=num_classes)
        elif self.model_type == 'EfficientNet':
            model = EfficientNetClassifier(num_classes=num_classes)
        elif self.model_type == 'ViT':
            model = VisionTransformer(num_classes=num_classes, depth=6, embed_dim=384)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Load weights
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        
        return model
    
    def preprocess_image(self, image_path):
        """Load and preprocess an image"""
        # Load image
        if isinstance(image_path, str) or isinstance(image_path, Path):
            image = Image.open(image_path).convert('RGB')
        else:
            image = image_path  # Assume it's already a PIL Image
        
        # Store original for visualization
        original_image = np.array(image)
        
        # Transform
        image_tensor = self.transform(image).unsqueeze(0)  # Add batch dimension
        
        return image_tensor, original_image
    
    def predict(self, image_path, return_probabilities=True):
        """
        Predict the class of a brain tumor image
        
        Args:
            image_path: Path to the image file or PIL Image
            return_probabilities: If True, return probabilities for all classes
            
        Returns:
            Dictionary containing prediction results
        """
        # Preprocess
        image_tensor, original_image = self.preprocess_image(image_path)
        image_tensor = image_tensor.to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
        
        # Prepare results
        results = {
            'predicted_class': self.class_names[predicted_class.item()],
            'predicted_class_index': predicted_class.item(),
            'confidence': confidence.item() * 100,
            'original_image': original_image
        }
        
        if return_probabilities:
            results['probabilities'] = {
                class_name: prob * 100 
                for class_name, prob in zip(self.class_names, probabilities[0].cpu().numpy())
            }
        
        return results
    
    def predict_batch(self, image_paths):
        """Predict multiple images at once"""
        results = []
        for img_path in image_paths:
            result = self.predict(img_path)
            result['image_path'] = str(img_path)
            results.append(result)
        return results
    
    def visualize_prediction(self, image_path, save_path=None, show_gradcam=True):
        """
        Visualize prediction with Grad-CAM
        
        Args:
            image_path: Path to the image
            save_path: Path to save the visualization (optional)
            show_gradcam: Whether to show Grad-CAM heatmap
        """
        # Get prediction
        results = self.predict(image_path)
        
        # Create visualization
        if show_gradcam and self.model_type in ['CustomCNN', 'ResNet50', 'EfficientNet']:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        else:
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Original image
        axes[0].imshow(results['original_image'])
        axes[0].set_title('Original MRI Scan', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # Prediction probabilities
        classes = list(results['probabilities'].keys())
        probs = list(results['probabilities'].values())
        colors = ['#ff6b6b' if i == results['predicted_class_index'] else '#4ecdc4' 
                  for i in range(len(classes))]
        
        axes[1].barh(classes, probs, color=colors)
        axes[1].set_xlabel('Confidence (%)', fontsize=12)
        axes[1].set_title('Class Probabilities', fontsize=14, fontweight='bold')
        axes[1].set_xlim(0, 100)
        
        # Add confidence values on bars
        for i, (cls, prob) in enumerate(zip(classes, probs)):
            axes[1].text(prob + 2, i, f'{prob:.1f}%', va='center', fontsize=10)
        
        # Grad-CAM visualization
        if show_gradcam and self.model_type in ['CustomCNN', 'ResNet50', 'EfficientNet']:
            try:
                # Get target layer for Grad-CAM
                if self.model_type == 'CustomCNN':
                    target_layer = self.model.conv4[-2]  # Last conv layer
                elif self.model_type == 'ResNet50':
                    target_layer = self.model.model.layer4[-1]
                elif self.model_type == 'EfficientNet':
                    target_layer = self.model.model.features[-1]
                
                # Generate Grad-CAM
                gradcam = GradCAM(self.model, target_layer)
                image_tensor, _ = self.preprocess_image(image_path)
                image_tensor = image_tensor.to(self.device)
                image_tensor.requires_grad = True
                
                heatmap, _ = gradcam.generate_cam(image_tensor, results['predicted_class_index'])
                
                # Overlay heatmap on original image
                heatmap_resized = cv2.resize(heatmap, (results['original_image'].shape[1], 
                                                       results['original_image'].shape[0]))
                heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
                heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
                
                superimposed = heatmap_colored * 0.4 + results['original_image'] * 0.6
                superimposed = np.clip(superimposed, 0, 255).astype(np.uint8)
                
                axes[2].imshow(superimposed)
                axes[2].set_title('Grad-CAM Visualization\n(Tumor Localization)', 
                                fontsize=14, fontweight='bold')
                axes[2].axis('off')
            except Exception as e:
                print(f"Could not generate Grad-CAM: {e}")
        
        # Add overall title
        prediction_text = f"Prediction: {results['predicted_class'].upper()} ({results['confidence']:.1f}% confidence)"
        fig.suptitle(prediction_text, fontsize=16, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Visualization saved to: {save_path}")
        
        plt.show()
        
        return results
