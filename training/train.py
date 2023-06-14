import mobilenetv3small, mobilenetv3large, resnet50, efficientnetv2small
from torchvision import datasets, models, transforms


efficientnetv2small.transfer_learning('datasets/bubble_resize', '/Users/brianprzezdziecki/Research/Mechatronics/My_code/Anomaly_Classification/models/best_model_transfer_large_efficient.pt', 3)

