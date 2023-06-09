import torch
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import numpy as np
from sklearn.model_selection import train_test_split
import os
from torch.utils.tensorboard import SummaryWriter

# TensorBoard setup
writer = SummaryWriter()

# Data loading and preprocessing
data_dir = 'datasets/classification_datasets'
train_transforms = transforms.Compose([
    transforms.RandomRotation(30),  # add random rotation
    transforms.ToTensor(),  # convert image to PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # normalize image data
])

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load datasets
dataset = datasets.ImageFolder(data_dir, transform=train_transforms)

# Split the dataset into training and validation datasets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

val_dataset.dataset.transform = test_transforms

# Create data loaders
batch_size = 4
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
# Model selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # use GPU if available
model = models.mobilenet_v2(pretrained=True)  # use pre-trained MobileNetV2 model
num_ftrs = model.classifier[1].in_features
model.classifier[1] = torch.nn.Linear(num_ftrs, 3)  # replace final layer to fit your problem
model = model.to(device)

# Training
criterion = torch.nn.CrossEntropyLoss()  # use cross entropy loss
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)  # use SGD optimizer
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)  # learning rate decay

# Early stopping initialization
n_epochs_stop = 5
min_val_loss = np.Inf
epochs_no_improve = 0

for epoch in range(25):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader, 0):
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
    scheduler.step()

    # validation loss
    val_loss = 0.0
    model.eval()  # Set model to evaluate mode
    for inputs, labels in val_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        val_loss += loss.item() * inputs.size(0)

    # calculate average losses
    running_loss /= len(train_loader.dataset)
    val_loss /= len(val_loader.dataset)

    # print training/validation statistics
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch, running_loss, val_loss))

    writer.add_scalars('Loss', {'train': running_loss, 'val': val_loss}, epoch)

    # early stopping and model saving
    if val_loss < min_val_loss:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
        min_val_loss,
        val_loss))
        torch.save(model.state_dict(), 'best_model.pt')
        epochs_no_improve = 0
        min_val_loss = val_loss
    else:
        epochs_no_improve += 1
    if epoch > 5 and epochs_no_improve == n_epochs_stop:
        print('Early stopping!')
        break
    else:
        continue
    break

writer.close()
print('Finished Training')

# Evaluation
correct = 0
total = 0
with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy on test images: %d %%' % (100 * correct / total))
