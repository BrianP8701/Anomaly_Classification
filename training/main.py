import train_model as train

'''
Path to folder containing dataset. The dataset should be organized as follows:
    dataset/
        train/
            class1/
                frame0.jpg
                ...
            class2/
                frame0.jpg
                ...
            ...
        val/
            ...
''' 
dataset = ''


# Choose which model you want from model_repository.py
model = ''


# Path to save weights of best model
destination_path = ''


'''
The metrics come in the following format:
    A list of lists:
        [accuracy, train_precisions, train_recalls, train_f1_scores, val_precisions, val_recalls, val_f1_scores]
        
    Accuracy is a single number in a list of length 0.
    The rest are lists of length num_epochs.
'''
# You may choose to use transfer learning or finetuning
model, metrics = train.transfer_learning('resnet152', data_dir=dataset, destination_path=destination_path, num_of_classes=3, batch_size=3, num_epochs=25)
#model, metrics = train.finetune('resnet152', data_dir=dataset, destination_path=destination_path, num_of_classes=3, batch_size=3, num_epochs=25)