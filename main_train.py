import training.train_model as train
import metrics.analyze_metrics as analyze

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

# Example input
dataset_path = 'datasets/active/gmms2'       # Path to dataset as described above
model_name = 'mobilenet_v3_large'         # Choose which model you want from model_list in model_repository.py
destination_path = 'models/best.pt'   # Path to save weights of best model


'''
The metrics come in the following format:
    A list of lists:
        [accuracy, train_precisions, train_recalls, train_f1_scores, val_precisions, val_recalls, val_f1_scores]
        
    Accuracy is a single number in a list of length 0.
    The rest are lists of length num_epochs.
'''

# Change the hyperparameters below as desired
model, metrics = train.train(model_name=model_name, data_dir=dataset_path, destination_path=destination_path, num_of_classes=3,
                             transfer_learning=False, batch_size=4, num_epochs=2, learning_rate=0.01, momentum=0.9, step_size=5, gamma=0.1)



# Path to JSON file to save metrics to (Create any empty JSON file and put the path here)
json_path = 'best.json'
# Key to save metrics for this trained model under. You can save multiple models to the same JSON file under different keys.
key = 'best_model'
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

analyze.add_data_to_json(json_path, key, sub_dict)