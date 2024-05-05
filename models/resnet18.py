import torch.nn as nn
import torch.optim as optim

from models.ots_models import get_model

class Resnet18(nn.Module):
    def __init__(self, num_classes, get_weights = False):
        super().__init__()
        self.model, _ = get_model("resnet18", get_weights)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        
    def forward(self, x):
        return self.model(x)
        