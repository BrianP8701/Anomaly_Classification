'''
    1. Select your model from the model_repository.py file
    2. Provide the path to your dataset. The dataset should be in the following format:

    dataset/
        class1/
            image1.jpg
            image2.jpg
            ...
        class2/
            image1.jpg
            image2.jpg
            ...
        ...
      
    3. Provide the path to where you want to save your model. The model will be saved as a .pt file.
    4. Provide hyperarameters for training. The default values are shown below.  
'''
import train
from torchvision import datasets, models, transforms

model = 'mobilenet_v3_small'
dataset = 'datasets/bubble_resize'
destination_path = '/Users/brianprzezdziecki/Research/Mechatronics/My_code/Anomaly_Classification/models/best_model_transfer_large_efficient.pt'

train.finetune(model=model, data_dir=dataset, destination_path=destination_path, num_of_classes=3, batch_size=4, num_epochs=25)

