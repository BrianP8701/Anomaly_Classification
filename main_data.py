import data_processing.dataset_processing as dataset_processing

'''    
    Dataset should be organized as follows:
    
        folder/
            class1/
                frame0.jpg
                frame1.jpg
                ...
            class2/
            ...
'''
dataset_path = 'datasets/backup/bubble'
destination_path = 'datasets/processed'

'''
    Adjust hyperparameters as needed.

    I'd recommend using the default values provided, and only changing the augmentations_per_class and flips_per_class values.
'''
dataset_processing.prepare_data_and_split(dataset_path, destination_path, augmentations_per_class=10, flips_per_class=10, 
                                          gmms=True, num_components=6, grayscale=True, resize=224, train_val_test_split=[0.7, 0.2, 0.1])