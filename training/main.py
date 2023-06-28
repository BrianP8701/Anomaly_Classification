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


# Choose which model you want from the following list:
# ['efficientnet_v2_s', 'efficientnet_v2_l', 'resnet18', 'resnet152', 'mobilenet_v3_small', 'mobilenet_v3_large']
model_name = ''


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
model, metrics = train.transfer_learning(model_name, data_dir=dataset, destination_path=destination_path, num_of_classes=3, batch_size=3, num_epochs=25)
#model, metrics = train.finetune('resnet152', data_dir=dataset, destination_path=destination_path, num_of_classes=3, batch_size=3, num_epochs=25)



# If you wish to save your metrics to a JSON file, you may uncomment the below lines:

# # Path to JSON file
# json_path = ''

# # Key to save metrics for this trained model under
# key = ''

# # Don't change this
# sub_dict = {
#             'accuracy': [metrics[0]],
#             'val_accuracy': [metrics[1]],
#             'train_precisions': metrics[2],
#             'train_recalls': metrics[3],
#             'train_f1_scores': metrics[4],
#             'val_precisions': metrics[5],
#             'val_recalls': metrics[6],
#             'val_f1_scores': metrics[7]
#         }
# train.add_data_to_json(json_path, key, sub_dict)