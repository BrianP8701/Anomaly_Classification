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

all_models = ['efficientnet_v2_s', 'efficientnet_v2_l', 'resnet18', 'resnet152', 'mobilenet_v3_small', 'mobilenet_v3_large']
all_datasets = ['datasets/50', 'datasets/224', 'datasets/224pad', 'datasets/gmms6', 'datasets/bubble_50', 'datasets/bubble_224', 'datasets/bubble_224pad']
model_abbreviations = ['eff_s', 'eff_l', 'res18', 'res152', 'mob_s', 'mob_l']

trained_models = []
accuracy_list = []

model_index = 0
for model in all_models:
    for dataset in all_datasets:
        
        destination_path = '/Users/brianprzezdziecki/Research/Mechatronics/My_code/Anomaly_Classification/models/' + model_abbreviations[model_index] + '_' + dataset.split('/')[1] + '.pt'
        model, accuracy = train.finetune(model=model, data_dir=dataset, destination_path=destination_path, num_of_classes=3, batch_size=4, num_epochs=25)
        trained_models.append(model_abbreviations[model_index] + '_' + dataset.split('/')[1])
        accuracy_list.append(accuracy)
        
    model_index += 1


for i in range(len(trained_models)):
    print(trained_models[i] + ': ' + str(accuracy_list[i]))