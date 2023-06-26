'''
    This file contains the code for training models.
    If you want to train a model, refer to the main.py file in this directory.

    The finetune and transfer_learning functions are functions you can call from another file.
    The train model function contains the code that does the training and evaluation.


    If there are updates in the future that require you to change the training code, you can do so here.
    If you want to change more hyperparameters, you can do so here.
'''

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import time
import os
from tempfile import TemporaryDirectory
from sklearn.metrics import precision_score, recall_score, f1_score
from model_repository import model_dict, weights_dict
from torchvision.transforms import functional as F
from sklearn.metrics import precision_recall_fscore_support
import json

def train_model(model, data_dir, destination_path, batch_size, num_epochs):
    cudnn.benchmark = True
    plt.ion()   # interactive mode
    
    # Data augmentation and normalization for training
    # Just normalization for validation
    # Your transformations
    data_transforms = {
        'train': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
        
    # Create datasets
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                            data_transforms[x])
                    for x in ['train', 'val']}
    # Create data loaders
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size,
                                                shuffle=True)
                for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Training
    criterion = torch.nn.CrossEntropyLoss()  # use cross entropy loss
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)  # use SGD optimizer

    # Decay LR by a factor of 0.1 every 5 epochs
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1) 
    
    # Initialize list for metrics
    train_precisions = [] 
    train_recalls = []
    train_f1_scores = []
    val_precisions = []
    val_recalls = []
    val_f1_scores = []

    since = time.time()

    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, destination_path)

        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)
            
            # At the beginning of each epoch, initialize lists to store all predictions and true labels
            all_preds = []
            all_labels = []

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                    
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    
                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    scheduler.step()
                
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
                
                # Calculate and save precision, recall, and f1 score
                precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)
                if(phase == 'train'):
                    train_precisions.append(precision)
                    train_recalls.append(recall)
                    train_f1_scores.append(f1)
                else:
                    val_precisions.append(precision)
                    val_recalls.append(recall)
                    val_f1_scores.append(f1)
                
                print(f'{phase} Precision: {precision:.4f} Recall: {recall:.4f} F1 score: {f1:.4f}')
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_params_path)
                    
            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path))
        
    return model, [best_acc.item(), train_precisions, train_recalls, train_f1_scores, val_precisions, val_recalls, val_f1_scores]

# Finetune weights of pretrained model on new dataset
def finetune(model_name, data_dir, destination_path, num_of_classes, batch_size=4, num_epochs=25):
    # Initialize model with pretrained weights
    weights = weights_dict[model_name]
    model = model_dict[model_name](weights=weights)
        
    # Reset final fully connected layer
    if(model_name == 'efficientnet_v2_s' or model_name == 'efficientnet_v2_m' or model_name == 'efficientnet_v2_l'):
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_of_classes)
    elif(model_name == 'mobilenet_v3_small' or model_name == 'mobilenet_v3_large'):
        num_ftrs = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(num_ftrs, num_of_classes)
    elif(model_name == 'resnet18' or model_name == 'resnet50' or model_name == 'resnet152'):
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_of_classes)

    model, metrics = train_model(model, data_dir, destination_path, batch_size, num_epochs)
    return model, metrics

# Freeze weights of pretrained model and train only the final fully connected layer
def transfer_learning(model_name, data_dir, destination_path, num_of_classes, batch_size=4, num_epochs=25):
    # Initialize model with pretrained weights
    weights = weights_dict[model_name]
    model = model_dict[model_name](weights=weights)
    
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False
        
    # Reset final fully connected layer
    if(model_name == 'efficientnet_v2_s' or model_name == 'efficientnet_v2_m' or model_name == 'efficientnet_v2_l'):
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_of_classes)
        # model.classifier[1].requires_grad = True
    elif(model_name == 'mobilenet_v3_small' or model_name == 'mobilenet_v3_large'):
        num_ftrs = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(num_ftrs, num_of_classes)
        # model.classifier[3].requires_grad = True
    elif(model_name == 'resnet18' or model_name == 'resnet50' or model_name == 'resnet152'):
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_of_classes)
        # model.fc.requires_grad = True
        
    model, metrics = train_model(model, data_dir, destination_path, batch_size, num_epochs)
    return model, metrics

'''
    This function adds a dictionary to a JSON file under a given key. If the file does not exist, it will be created.
    
    In context of this project, the JSON file is used to store metrics for each model. The key is the model name and the value is a dictionary of metrics.
'''
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