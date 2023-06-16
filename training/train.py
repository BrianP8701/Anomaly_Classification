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
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
from PIL import Image
from tempfile import TemporaryDirectory
from sklearn.metrics import precision_score, recall_score, f1_score
from model_repository import model_dict, weights_dict

def train_model(model, data_dir, destination_path, batch_size, num_epochs):
    cudnn.benchmark = True
    plt.ion()   # interactive mode
    
    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(30),  # add random rotation
            transforms.ToTensor(),  # convert image to PyTorch tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # normalize image data
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    
    # Load datasets
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

    since = time.time()
    
    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, destination_path)

        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0
        # y_true = []
        # y_pred = []
        # precision_list = []
        # recall_list = []
        # f1_list = []

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

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
                        
                        # # Append the true labels and predictions to your lists
                        # y_true.extend(labels.cpu().numpy())
                        # y_pred.extend(preds.cpu().numpy())

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    scheduler.step()
                    
                # # Calculate metrics
                # precision = precision_score(y_true, y_pred, average='weighted')
                # recall = recall_score(y_true, y_pred, average='weighted')
                # f1 = f1_score(y_true, y_pred, average='weighted')
                
                # precision_list.append(precision)
                # recall_list.append(recall)
                # f1_list.append(f1)
                
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_params_path)
                    
            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')
        # print(f'Precision: {precision_list}')
        # print(f'Recall: {recall_list}')
        # print(f'F1: {f1_list}')

        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path))
        
    return model, best_acc

# Finetune weights of pretrained model on new dataset
def finetune(model, data_dir, destination_path, num_of_classes, batch_size=4, num_epochs=25):
    # Initialize model with pretrained weights
    weights = weights_dict[model]
    model = model_dict[model](weights=weights)
        
    # Reset final fully connected layer
    if(model == 'efficientnet_v2_s' or model == 'efficientnet_v2_m' or model == 'efficientnet_v2_l'):
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_of_classes)
    elif(model == 'mobilenet_v3_small' or model == 'mobilenet_v3_large'):
        num_ftrs = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(num_ftrs, num_of_classes)
    elif(model == 'resnet18' or model == 'resnet50' or model == 'resnet152'):
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_of_classes)

    model, accuracy = train_model(model, data_dir, destination_path, batch_size, num_epochs)
    return model, accuracy

# Freeze weights of pretrained model and train only the final fully connected layer
def transfer_learning(model, data_dir, destination_path, num_of_classes, batch_size=4, num_epochs=25):
    # Initialize model with pretrained weights
    weights = weights_dict[model]
    model = model_dict[model](weights=weights)
    
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False
        
    # Reset final fully connected layer
    if(model == 'efficientnet_v2_s' or model == 'efficientnet_v2_m' or model == 'efficientnet_v2_l'):
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_of_classes)
    elif(model == 'mobilenet_v3_small' or model == 'mobilenet_v3_large'):
        num_ftrs = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(num_ftrs, num_of_classes)
    elif(model == 'resnet18' or model == 'resnet50' or model == 'resnet152'):
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_of_classes)
        
    model, accuracy = train_model(model, data_dir, destination_path, batch_size, num_epochs)
    return model, accuracy