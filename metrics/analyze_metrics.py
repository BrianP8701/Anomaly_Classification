'''
    This script is used to analyze the metrics of a model and plot them.
'''
import json
import matplotlib.pyplot as plt
import os

''' 
    Given the metrics for one model, in the form of a dictionary, plot the metrics
    
    model_data: a dictionary of the form:
        {
            metric1: [scores]
            metric2: [scores]
        }
'''
def plot_metrics(model_data, model_name, destination_path, metrics=['train_precisions', 'train_recalls', 'train_f1_scores', 'val_precisions', 'val_recalls', 'val_f1_scores']):
    fig, axs = plt.subplots(6, figsize=(10, 20))
    fig.suptitle(f'Model: {model_name} - Accuracy: {model_data["accuracy"][0]}', fontsize=16)

    for i, metric in enumerate(metrics):
        axs[i].plot(model_data[metric])
        axs[i].set_title(metric)
        axs[i].set_xlabel('Iteration')
        axs[i].set_ylabel('Score')

    plt.tight_layout()
    fig.subplots_adjust(top=0.94)
    plt.savefig(destination_path)
    plt.close()

''' 
    Given the metrics for multiple models, in the form of a dictionary, plot their metrics in one figure
    
    models_data: a dictionary of the form:
        {
            model_name: {metrics}
            model_name: {metrics}
            ...
        }
'''
def plot_multiple_models(models_data, destination_path):
    metrics = ['accuracy', 'val_accuracy', 'train_precisions', 'train_recalls', 'train_f1_scores', 'val_precisions', 'val_recalls', 'val_f1_scores']
    fig, axs = plt.subplots(len(metrics), figsize=(10, 20))

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    color_index = 0

    for model_name, model_data in models_data.items():
        for i, metric in enumerate(metrics):
            if metric == 'accuracy':
                axs[i].scatter([0], model_data[metric], color=colors[color_index], label=model_name)
                axs[i].set_title(metric)
                axs[i].set_xlabel('Iteration')
                axs[i].set_ylabel('Score')
                axs[i].legend()
            elif metric == 'val_accuracy':
                axs[i].scatter([0], model_data[metric], color=colors[color_index], label=model_name)
                axs[i].set_title(metric)
                axs[i].set_xlabel('Iteration')
                axs[i].set_ylabel('Score')
                axs[i].legend()
            else:
                axs[i].plot(model_data[metric], color=colors[color_index], label=model_name)
                axs[i].set_title(metric)
                axs[i].set_xlabel('Iteration')
                axs[i].set_ylabel('Score')
                axs[i].legend()
        color_index = (color_index + 1) % len(colors)

    plt.tight_layout()
    fig.subplots_adjust(top=0.94)
    plt.savefig(destination_path)
    plt.close()


def calculate_range(scores):
    # return approximate range
    return {
        "min": min(scores),
        "max": max(scores)
    }


# This file accepts metrics, and creates a new JSON file with the metrics in a different format: The ranges of the scores
def process_file(filename):
    with open(filename, 'r') as f:
        data = json.load(f)

    new_data = {}
    for model, metrics in data.items():
        new_model_data = {
            "accuracy": metrics['accuracy'],
            "train_precisions": calculate_range(metrics['train_precisions']),
            "train_recalls": calculate_range(metrics['train_recalls']),
            "train_f1_scores": calculate_range(metrics['train_f1_scores']),
            "val_precisions": calculate_range(metrics['val_precisions']),
            "val_recalls": calculate_range(metrics['val_recalls']),
            "val_f1_scores": calculate_range(metrics['val_f1_scores'])
        }
        new_data[model] = new_model_data

    with open('new_file.json', 'w') as f:
        json.dump(new_data, f, indent=4)


# Given a JSON file and string (Called flag), move all key-value pairs with a key that contains the flag string to a new JSON file
def move_bubble_keys(src_path, dst_path, flag):
    # Open and read the source JSON file
    with open(src_path, 'r') as src_file:
        src_dict = json.load(src_file)

    # Initialize an empty dictionary to store key-value pairs with 'bubble'
    bubble_dict = {}
    keys_to_remove = []

    # Loop through keys in source dictionary
    for key in src_dict.keys():
        # If key contains 'bubble', add the key-value pair to bubble_dict and store the key for removal
        if flag in key:
            bubble_dict[key] = src_dict[key]
            keys_to_remove.append(key)

    # Remove the 'bubble' keys from the source dictionary
    for key in keys_to_remove:
        del src_dict[key]

    # Write the bubble_dict to the destination JSON file
    with open(dst_path, 'w') as dst_file:
        json.dump(bubble_dict, dst_file)

    # Overwrite the source JSON file with the modified dictionary
    with open(src_path, 'w') as src_file:
        json.dump(src_dict, src_file)
        
        
# Given 2 JSON files, combine their key-value pairs into a new JSON file
def combine_json(file1_path, file2_path, output_path):
    # Open the first file and load the data
    with open(file1_path, 'r') as file1:
        data1 = json.load(file1)

    # Open the second file and load the data
    with open(file2_path, 'r') as file2:
        data2 = json.load(file2)

    # Combine the data from both files
    combined_data = {**data1, **data2}

    # Write the combined data to the output file
    with open(output_path, 'w') as output_file:
        json.dump(combined_data, output_file)

# Given list of attributes, return a list of means of those attributes
def get_means(json_path, attributes):
    with open(json_path, 'r') as src_file:
        data = json.load(src_file)
        
    means = []
    val_means = []
    
    for attribute in attributes:
        test_mean = 0
        val_mean = 0
        count = 0
        
        for model in data.keys():
            if any(s in model for s in attribute):
                print(f'{attribute} in {model}')
                test_mean += data[model]['accuracy'][0]
                print(test_mean)
                print()
                val_mean += data[model]['val_accuracy'][0]
                count += 1
                
        means.append([attribute, test_mean / count])
        val_means.append([attribute, val_mean / count])
        
    return ['test', means], ['val', val_means]


def add_data_to_json(filename, key, sub_dict):
    '''
    This function adds a dictionary to a JSON file under a given key. If the file does not exist, it will be created.
    
    In context of this project, the JSON file is used to store metrics for each model. The key is the model name and the value is a dictionary of metrics.
    '''
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