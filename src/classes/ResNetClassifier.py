import torch.nn as nn
from torchvision import models

class ResNetClassifier(nn.Module):
    """ResNet-based Classifier"""
    def __init__(self, num_classes=4, pretrained=True):
        super(ResNetClassifier, self).__init__()
        self.model = models.resnet50(pretrained=pretrained)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        
    def forward(self, x):
        return self.model(x)