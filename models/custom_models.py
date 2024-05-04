from models.vit import ViT

def get_model(num_classes, model_name = 'vit'):
    model = {}
    if model_name == "vit":
        model = ViT(num_classes)
    else:
        raise SystemExit("Error: no valid model name passed! Check run.yaml")
    return model