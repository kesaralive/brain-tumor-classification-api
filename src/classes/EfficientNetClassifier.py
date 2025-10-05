import torch.nn as nn
from torchvision import models

class EfficientNetClassifier(nn.Module):
    """EfficientNet-based Classifier"""
    def __init__(self, num_classes=4, pretrained=True):
        super(EfficientNetClassifier, self).__init__()
        self.model = models.efficientnet_b0(pretrained=pretrained)
        self.model.classifier[1] = nn.Linear(
            self.model.classifier[1].in_features, num_classes
        )
        
    def forward(self, x):
        return self.model(x)
