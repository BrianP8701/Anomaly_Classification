import json
import matplotlib.pyplot as plt

# Given the metrics for one model, in the form of a dictionary, plot the metrics
def plot_metrics(model_data, model_name, destination_path):
    fig, axs = plt.subplots(6, figsize=(10, 20))
    fig.suptitle(f'Model: {model_name} - Accuracy: {model_data["accuracy"][0]}', fontsize=16)

    metrics = ['train_precisions', 'train_recalls', 'train_f1_scores', 'val_precisions', 'val_recalls', 'val_f1_scores']
    for i, metric in enumerate(metrics):
        axs[i].plot(model_data[metric])
        axs[i].set_title(metric)
        axs[i].set_xlabel('Iteration')
        axs[i].set_ylabel('Score')

    plt.tight_layout()
    fig.subplots_adjust(top=0.94)
    plt.savefig(destination_path)
    plt.close()

# Given the metrics for multiple models, in the form of a dictionary, plot their metrics in one figure
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

def move_bubble_keys(src_path, dst_path):
    # Open and read the source JSON file
    with open(src_path, 'r') as src_file:
        src_dict = json.load(src_file)

    # Initialize an empty dictionary to store key-value pairs with 'bubble'
    bubble_dict = {}
    keys_to_remove = []

    # Loop through keys in source dictionary
    for key in src_dict.keys():
        # If key contains 'bubble', add the key-value pair to bubble_dict and store the key for removal
        if 'bubble' in key:
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

with open('metrics/bubble_metrics.json', 'r') as src_file:
    data = json.load(src_file)
        
max_accuracy = 0
max_accuracy_model = ''
for model_name, model_data in data.items():
    if model_data['accuracy'][0] > max_accuracy:
        max_accuracy = model_data['accuracy'][0]
        max_accuracy_model = model_name
        
print(max_accuracy_model, max_accuracy)