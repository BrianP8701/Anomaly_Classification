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
from model_repository import model_dict, weights_dict
import json
import os

def add_data_to_json(filename, key, sub_dict):
    try:
        # If the file exists, load its data. Otherwise, start with an empty dict.
        if os.path.isfile(filename):
            with open(filename, 'r') as f:
                data = json.load(f)
        else:
            data = {}
    except json.decoder.JSONDecodeError:
        data = {}  # start with an empty dictionary if file is empty

    # Add the new dictionary to the data under the given key
    data[key] = sub_dict

    # Write the updated data back to the file
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

all_models = ['efficientnet_v2_s', 'efficientnet_v2_l', 'resnet18', 'resnet152', 'mobilenet_v3_small', 'mobilenet_v3_large']
all_datasets = ['datasets/bubble', 'datasets/bubble_pad', 'datasets/bubble_resize', 'datasets/classification', 'datasets/gmms6', 'datasets/pad', 'datasets/resize']
model_abbreviations = ['eff_s', 'eff_l', 'res18', 'res152', 'mob_s', 'mob_l']

model_index = 0
# Loop through all models
for model_name in all_models:
    # Loop through all datasets
    for dataset in all_datasets:
        print('\033[92m' + model_name + ' ' + dataset + '\033[0m') 
        
        key = model_abbreviations[model_index] + '_' + dataset.split('/')[1] + '_transfer'
        destination_path = '/Users/brianprzezdziecki/Research/Mechatronics/My_code/Anomaly_Classification/models/' + key + '.pt'
        model, metrics = train.transfer_learning(model_name=model_name, data_dir=dataset, destination_path=destination_path, num_of_classes=3, batch_size=3, num_epochs=25)
        sub_dict = {
            'accuracy': [metrics[0]],
            'train_precisions': metrics[1],
            'train_recalls': metrics[2],
            'train_f1_scores': metrics[3],
            'val_precisions': metrics[4],
            'val_recalls': metrics[5],
            'val_f1_scores': metrics[6]
        }
        add_data_to_json('models/metrics.json', key, sub_dict)
        
        key = model_abbreviations[model_index] + '_' + dataset.split('/')[1] + '_finetune'
        destination_path = '/Users/brianprzezdziecki/Research/Mechatronics/My_code/Anomaly_Classification/models/' + key + '.pt'
        model, metrics = train.finetune(model_name=model_name, data_dir=dataset, destination_path=destination_path, num_of_classes=3, batch_size=3, num_epochs=25)
        
        sub_dict = {
            'accuracy': [metrics[0]],
            'train_precisions': metrics[1],
            'train_recalls': metrics[2],
            'train_f1_scores': metrics[3],
            'val_precisions': metrics[4],
            'val_recalls': metrics[5],
            'val_f1_scores': metrics[6]
        }
        add_data_to_json('models/metrics.json', key, sub_dict)
        
    model_index += 1