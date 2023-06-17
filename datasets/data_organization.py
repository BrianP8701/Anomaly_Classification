'''
Provides methods to organize data to work with the training methods in the repository.
'''

import os
import shutil
from sklearn.model_selection import train_test_split
import random

"""
    Splits each class in the given data directory into training and validation subsets.

    Args:
        data_dir (str): The directory containing the data. This directory should have a subdirectory for each class,
                        and each subdirectory should contain the images for that class.
                        For example:

                        data_dir/
                            class1/
                                image1.jpg
                                image2.jpg
                                ...
                            class2/
                                image1.jpg
                                image2.jpg
                                ...
                            ...
        val_size (float): The proportion of images from each class to put into the validation set. This should be a 
                          decimal between 0 and 1. For example, 0.2 means 20% of images from each class will be used 
                          for validation.

    The function will rearrange the images in the data directory such that it has a separate subdirectory for training
    and validation sets. Each of these subdirectories will then have a subdirectory for each class. For example:

        data_dir/
            train/
                class1/
                class2/
                ...
            val/
                class1/
                class2/
                ...
"""
def split_into_train_val(data_dir, val_size):
    # Ensure val_size is a valid proportion
    assert 0 <= val_size <= 1, "val_size should be a decimal between 0 and 1"

    # Loop through each class in the data_dir
    for class_name in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, class_name)

        # Skip if it's not a directory (for example, .DS_Store files on MacOS)
        if not os.path.isdir(class_dir):
            continue

        # Collect all image files in the class_dir
        image_files = [f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))]

        # Shuffle image_files for randomness
        random.shuffle(image_files)

        # Calculate the number of validation files
        num_val_files = int(len(image_files) * val_size)

        # Get the validation files
        val_files = image_files[:num_val_files]

        # Create validation directory if it doesn't exist
        val_dir = os.path.join(data_dir, 'val', class_name)
        if not os.path.exists(val_dir):
            os.makedirs(val_dir)

        # Move the validation files to the validation directory
        for val_file in val_files:
            shutil.move(os.path.join(class_dir, val_file), val_dir)

        # Create training directory if it doesn't exist
        train_dir = os.path.join(data_dir, 'train', class_name)
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)

        # Move the remaining files to the training directory
        for train_file in os.listdir(class_dir):
            shutil.move(os.path.join(class_dir, train_file), train_dir)

        # Delete the original class directory
        os.rmdir(class_dir)
        
        # Call rename_files on both the train and val directories for this class
        rename_files(train_dir)
        rename_files(val_dir)
                
'''            
If you have a folder of images formatted like this:
    frame5.jpg
    frame7.jpg
    frame13.jpg
    ...
You can use this method to order them into:
    frame0.jpg
    frame1.jpg
    frame2.jpg
    ...
'''
def rename_files(directory_path):
    # Get a list of all files in the directory
    files = os.listdir(directory_path)
    
    # Sort the list of files based on the number in the file name
    files.sort(key=lambda f: int(f.split('frame')[1].split('.jpg')[0]))
    
    # Iterate over the sorted list and rename each file
    for i, file in enumerate(files):
        old_path = os.path.join(directory_path, file)
        new_path = os.path.join(directory_path, f'frame{i}.jpg')
        os.rename(old_path, new_path)


source_dir = 'datasets/resize'
val_size = 0.2  # 20% of the data will be in the validation set

split_into_train_val(source_dir, val_size)