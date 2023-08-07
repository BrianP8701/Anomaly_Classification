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
import train_model as train
from torchvision import datasets, models, transforms
from model_repository import model_dict, weights_dict

# NOTE - eff to mob || res to eff || mob to res

all_models = ['mobilenet_v3_small', 'mobilenet_v3_large']
all_datasets = ['datasets/active/gmms', 'datasets/active/original', 'datasets/active/gmms2', 'datasets/active/original2']
dataset_abbreviations = ['gmms', 'original', 'gmms2', 'original2']
model_abbreviations = ['mob_s', 'mob_l']
json_path = 'metrics/metrics.json'

model_index = 0
# Loop through all models
for model_name in all_models:
    dataset_index = 0
    # Loop through all datasets
    for dataset in all_datasets:
        print('\033[92m' + model_name + ' ' + dataset + '\033[0m') 
        
        # Train model
        key = model_abbreviations[model_index] + '_' + dataset_abbreviations[dataset_index] + '_finetune'
        destination_path = '/Users/brianprzezdziecki/Research/Mechatronics/My_code/Anomaly_Classification/models/' + key + '.pt'
        model, metrics = train.finetune(model_name=model_name, data_dir=dataset, destination_path=destination_path, num_of_classes=3, batch_size=5, num_epochs=13)
        
        sub_dict = {
            'accuracy': [metrics[0]],
            'val_accuracy': [metrics[1]],
            'train_precisions': metrics[2],
            'train_recalls': metrics[3],
            'train_f1_scores': metrics[4],
            'val_precisions': metrics[5],
            'val_recalls': metrics[6],
            'val_f1_scores': metrics[7]
        }
        train.add_data_to_json(json_path, key, sub_dict)
         
        dataset_index += 1
        
    model_index += 1
