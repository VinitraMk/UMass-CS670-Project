import torchvision

def get_model(model_name = "resnet18", get_weights = False):
    model = {}
    model_id2name = {}
    if model_name.lower() == "resnet18":
        if get_weights:
            weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
            model = torchvision.models.resnet18(weights=weights)
            model_id2name = weights.meta["categories"]
        else:
            model = torchvision.models.resnet18()
    elif model_name.lower() == "inception":
        if get_weights:
            weights = torchvision.models.Inception_V3_Weights.IMAGENET1K_V1
            model = torchvision.models.inception_v3(weights=weights)
            model_id2name = weights.meta["categories"]
        else:
            model = torchvision.models.inception_v3()
    elif model_name.lower() == "vgg":
        if get_weights:
            weights = torchvision.models.VGG19_Weights.IMAGENET1K_V1
            model = torchvision.models.vgg19(weights=weights)
            model_id2name = weights.meta["categories"]
        else:
            model = torchvision.models.vgg19()
    elif model_name.lower() == "squeezenet":
        if get_weights:
            weights = torchvision.models.SqueezeNet1_1_Weights.IMAGENET1K_V1
            model = torchvision.models.squeezenet1_1(weights = weights)
            model_id2name = weights.meta["categories"]
        else:
            model = torchvision.models.squeezenet1_1()
    elif model_name.lower() == "vit":
        if get_weights:
            weights = torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1
            model = torchvision.models.vit_b_16(weights = weights)
            model_id2name = weights.meta["categories"]
        else:
            model = torchvision.models.vit_b_16()
    return model, model_id2name