'''
    This file contains the code for training models.
    If you want to train a model, refer to the main.py file in this directory.

    The finetune and transfer_learning functions are functions you can call from another file.
    The train model function contains the code that does the training and evaluation.


    If there are updates in the future that require you to change the training code, you can do so here.
    If you want to change more hyperparameters, you can do so here.
'''

from model_repository import model_dict, weights_dict
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
from torchvision.transforms import functional as F
from sklearn.metrics import precision_recall_fscore_support
import json



def train(model_name, data_dir, destination_path, num_of_classes, transfer_learning=False, batch_size=4, num_epochs=25, learning_rate=0.01, momentum=0.9, step_size=5, gamma=0.1):
    if transfer_learning:
        model, metrics = transfer_learning(model_name, data_dir, destination_path, num_of_classes, batch_size, num_epochs, learning_rate, momentum, step_size, gamma)
    else:
        model, metrics = finetune(model_name, data_dir, destination_path, num_of_classes, batch_size, num_epochs, learning_rate, momentum, step_size, gamma)
    return model, metrics



# Finetune weights of pretrained model on new dataset
def finetune(model_name, data_dir, destination_path, num_of_classes, batch_size=4, num_epochs=25, learning_rate=0.01, momentum=0.9, step_size=5, gamma=0.1):
    # Initialize model with pretrained weights
    weights = weights_dict[model_name]
    model = model_dict[model_name](weights=weights)
        
    # Reset final fully connected layer
    if 'efficient' in model_name:
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_of_classes)
    elif 'mobile' in model_name:
        num_ftrs = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(num_ftrs, num_of_classes)
    elif 'res' in model_name:
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_of_classes)

    model, metrics = train_model(model, data_dir, destination_path, batch_size, num_epochs, learning_rate, momentum, step_size, gamma)
    return model, metrics



# Freeze weights of pretrained model and train only the final fully connected layer
def transfer_learning(model_name, data_dir, destination_path, num_of_classes, batch_size=4, num_epochs=25, learning_rate=0.01, momentum=0.9, step_size=5, gamma=0.1):
    # Initialize model with pretrained weights
    weights = weights_dict[model_name]
    model = model_dict[model_name](weights=weights)
    
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False
        
    # Reset final fully connected layer
    if 'efficient' in model_name:
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_of_classes)
    elif 'mobile' in model_name:
        num_ftrs = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(num_ftrs, num_of_classes)
    elif 'res' in model_name:
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_of_classes)
        
    model, metrics = train_model(model, data_dir, destination_path, batch_size, num_epochs, learning_rate, momentum, step_size, gamma)
    return model, metrics



def train_model(model, data_dir, destination_path, batch_size, num_epochs, learning_rate=0.01, momentum=0.9, step_size=5, gamma=0.1):
    cudnn.benchmark = True
    plt.ion()   # interactive mode
    
    # Data augmentation and normalization for training
    # Just normalization for validation and testing
    data_transforms = {
        'train': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
        
    # Create datasets
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                            data_transforms[x])
                    for x in ['train', 'val', 'test']}
    # Create data loaders
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size,
                                                shuffle=True)
                for x in ['train', 'val', 'test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
    class_names = image_datasets['train'].classes
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Training
    criterion = torch.nn.CrossEntropyLoss()  # use cross entropy loss
    optimizer = torch.optim.SGD(model.parameters(), learning_rate, momentum)  # use SGD optimizer

    # Decay LR by a factor of 0.1 every 5 epochs
    scheduler = lr_scheduler.StepLR(optimizer, step_size, gamma) 

    # Initialize list for metrics
    train_precisions = [] 
    train_recalls = []
    train_f1_scores = []
    val_precisions = []
    val_recalls = []
    val_f1_scores = []

    since = time.time()

    # Create a temporary directory to save training checkpoints
    best_model_params_path = destination_path  # Directly use the provided destination_path

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

    # Final evaluation on the test set
    model.eval()   # Set model to evaluate mode
    running_corrects = 0
    for inputs, labels in dataloaders['test']:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # forward
        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

        # statistics
        running_corrects += torch.sum(preds == labels.data)

    test_acc = running_corrects.double() / dataset_sizes['test']
    print(f'Test Acc: {test_acc:4f}')
        
    return model, [test_acc.item(), best_acc.item(), train_precisions, train_recalls, train_f1_scores, val_precisions, val_recalls, val_f1_scores]