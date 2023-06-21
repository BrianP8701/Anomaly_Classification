import json
import matplotlib.pyplot as plt

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
    metrics = ['accuracy', 'train_precisions', 'train_recalls', 'train_f1_scores', 'val_precisions', 'val_recalls', 'val_f1_scores']
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
        
with open('metrics/metrics.json', 'r') as f:
        data = json.load(f)

models_data = {
    'res152_gmms6_transfer': data['res152_gmms6_transfer'],
    'res152_gmms6_finetune': data['res152_gmms6_finetune'],
    'res152_classification_transfer': data['res152_classification_transfer'],
    'res152_classification_finetune': data['res152_classification_finetune']
}

# Given a JSON file, move all key-value pairs with a key that contains the flag string to a new JSON file
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

with open('metrics/metrics_bubble.json', 'r') as src_file:
    data = json.load(src_file)
        
max_accuracy = 0
worst_accuracy = 1
max_accuracy_model = ''
worst_accuracy_model = ''
sum = 0
count = 0
for model_name, model_data in data.items():
    if 'transfer' in model_name:
        sum += model_data['accuracy'][0]
        count+=1
    print(model_name, model_data['accuracy'][0])
    if model_data['accuracy'][0] > max_accuracy:
        max_accuracy = model_data['accuracy'][0]
        max_accuracy_model = model_name
    if model_data['accuracy'][0] < worst_accuracy:
        worst_accuracy = model_data['accuracy'][0]
        worst_accuracy_model = model_name
        
print(sum/count)

print()
print(max_accuracy_model, max_accuracy)
print(worst_accuracy_model, worst_accuracy)
# all_models = ['efficientnet_v2_s', 'efficientnet_v2_l', 'resnet18', 'resnet152', 'mobilenet_v3_small', 'mobilenet_v3_large']
# all_datasets = ['datasets/bubble', 'datasets/bubble_pad', 'datasets/bubble_resize', 'datasets/classification', 'datasets/gmms6', 'datasets/pad', 'datasets/resize']
# model_abbreviations = ['eff_s', 'eff_l', 'res18', 'res152', 'mob_s', 'mob_l']

# model_index = 0
# for model in all_models:
#     all_keys = []
#     for dataset in all_datasets:
#         key = model_abbreviations[model_index] + '_' + dataset.split('/')[1] + '_finetune'
#         all_keys.append(key)
#     sub_dict = {}
#     for key in all_keys:
#         sub_dict[key] = data[key]
#     plot_multiple_models(sub_dict, f'plots/finetune/{model}_finetune.png')
#     model_index += 1
    
