from models.vit import ViT
from models.resnet18 import Resnet18

def get_model(num_classes, model_name = 'vit'):
    model = {}
    if model_name == "vit":
        model = ViT(num_classes, True)
    elif model_name == "resnet18":
        model = Resnet18(num_classes, True)
    else:
        raise SystemExit("Error: no valid model name passed! Check run.yaml")
    return model