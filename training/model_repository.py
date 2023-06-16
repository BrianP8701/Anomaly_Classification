'''
    This file simply contains a dictionary of torchvision 
    models and their weights for easy access.
'''
from torchvision import models

model_dict = {
    "efficientnet_v2_s": models.efficientnet_v2_s,
    "efficientnet_v2_l": models.efficientnet_v2_l,
    "resnet18": models.resnet18,
    "resnet152": models.resnet152,
    "mobilenet_v3_small": models.mobilenet_v3_small,
    "mobilenet_v3_large": models.mobilenet_v3_large,
}

weights_dict = {
    "efficientnet_v2_s": models.EfficientNet_V2_S_Weights.DEFAULT,
    "efficientnet_v2_l": models.EfficientNet_V2_L_Weights.DEFAULT,
    "resnet18": models.ResNet18_Weights.DEFAULT,
    "resnet152": models.ResNet152_Weights.DEFAULT,
    "mobilenet_v3_small": models.MobileNet_V3_Small_Weights.DEFAULT,
    "mobilenet_v3_large": models.MobileNet_V3_Large_Weights.DEFAULT,
}