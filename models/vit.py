import torch.nn as nn
import torch.optim as optim

from models.ots_models import get_model

class ViT(nn.Module):
    def __init__(self, num_classes, get_weights = False):
        super().__init__()
        self.model, _ = get_model("vit", get_weights)
        self.model.heads.head = nn.Linear(self.model.heads.head.in_features, num_classes)
        
    def forward(self, x):
        return self.model(x)
        