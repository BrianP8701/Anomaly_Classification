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

all_models = ['mobilenet_v3_small', 'mobilenet_v3_large', 'resnet18', 'resnet152']
all_datasets = ['datasets/gmms_224', 'datasets/original', 'datasets/gmms_actually50', 'datasets/gmms_50', 'datasets/resize']
model_abbreviations = ['mob_s', 'mob_l', 'res18', 'res152']
json_path = 'metrics/metrics.json'

model_index = 0
# Loop through all models
for model_name in all_models:
    # Loop through all datasets
    for dataset in all_datasets:
        print('\033[92m' + model_name + ' ' + dataset + '\033[0m') 
        
        key = model_abbreviations[model_index] + '_' + dataset.split('/')[1] + '_transfer'
        destination_path = '/Users/brianprzezdziecki/Research/Mechatronics/My_code/Anomaly_Classification/models/' + key + '.pt'
        model, metrics = train.transfer_learning(model_name=model_name, data_dir=dataset, destination_path=destination_path, num_of_classes=3, batch_size=3, num_epochs=13)
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
        
        
        
        key = model_abbreviations[model_index] + '_' + dataset.split('/')[1] + '_finetune'
        destination_path = '/Users/brianprzezdziecki/Research/Mechatronics/My_code/Anomaly_Classification/models/' + key + '.pt'
        model, metrics = train.finetune(model_name=model_name, data_dir=dataset, destination_path=destination_path, num_of_classes=3, batch_size=3, num_epochs=13)
        
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
        
    model_index += 1