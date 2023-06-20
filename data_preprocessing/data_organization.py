'''
Provides methods to organize files and folders to work with the training methods in this repository.
'''

import os
import shutil
from sklearn.model_selection import train_test_split
import random
from PIL import Image
import hashlib
import cv2
import numpy as np
import shutil

"""
    This function organizes .jpg files into their respective class folders.
    
    Parameters:
    folder_list (list of str): A list of strings, each string is a path to a source folder.
    destination_folder (str): A path to the main destination folder.
    class_list (list of str): A list of classes corresponding to subfolder names.
    
    Each source folder should have subfolders for each class, and each subfolder should contain the images for that class.
    The names of each subfolder should be the same across all source folders, and match the class_list.
        
    Usage:
    organize_data(["data/data13", "data/data14", "data/data15"], "maindata", ["normal", "over", "under"])
"""
def organize_data(folder_list, destination_folder, class_list):
    # Initialize a counter for each class
    counters = {class_name: 0 for class_name in class_list}

    # Loop through each source folder
    for source_folder in folder_list:
        # For each class
        for class_name in class_list:
            # Construct the source path
            source_path = os.path.join(source_folder, class_name)

            # Construct the destination path
            destination_path = os.path.join(destination_folder, class_name)

            # Create the destination path if it doesn't exist
            if not os.path.exists(destination_path):
                os.makedirs(destination_path)

            # Loop through each file in the source path
            for filename in os.listdir(source_path):
                # Only move .jpg files
                if filename.endswith(".jpg"):
                    # Construct the source file path
                    source_file_path = os.path.join(source_path, filename)

                    # Get the correct counter for this class
                    counter = counters[class_name]

                    # Construct the destination file path, with the new filename
                    destination_file_path = os.path.join(destination_path, f"frame{counter}.jpg")

                    # Move the file
                    shutil.move(source_file_path, destination_file_path)

                    # Increment the counter
                    counters[class_name] += 1

                    print(f"Moved {source_file_path} to {destination_file_path}")

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
        rename_images(train_dir)
        rename_images(val_dir)
        
'''            
If you have a folder of images, use this method to order them into:
    frame0.jpg
    frame1.jpg
    frame2.jpg
    ...
'''
def rename_images(folder_path):
    # Get the list of all files
    files = os.listdir(folder_path)

    # Sort them by creation time
    files.sort(key=lambda x: os.path.getmtime(os.path.join(folder_path, x)))

    # Initialize a counter
    counter = 0

    for filename in files:
        # Check if the file is a .jpg image
        if filename.endswith('.jpg'):
            # Construct the new name
            new_filename = 'frame{}.jpg'.format(counter)

            # If the file is already correctly named, skip it
            if filename == new_filename:
                counter += 1
                continue

            # Construct the full destination path
            destination = os.path.join(folder_path, new_filename)

            # Check if a file with the new name already exists
            while os.path.exists(destination):
                # If it does, increment the counter and construct a new name
                counter += 1
                new_filename = 'frame{}.jpg'.format(counter)
                destination = os.path.join(folder_path, new_filename)

            # Construct the full source path
            source = os.path.join(folder_path, filename)

            # Rename the file
            os.rename(source, destination)

            # Increment the counter
            counter += 1

def remove_duplicates(folder_path):
    # Get the list of all files
    files = os.listdir(folder_path)

    # Initialize a list to store image hashes
    hashes = []

    for filename in files:
        # Check if the file is a .jpg image
        if filename.endswith('.jpg'):
            # Open the image
            with Image.open(os.path.join(folder_path, filename)) as img:
                # Calculate the hash of the image and convert it to a hexadecimal string
                image_hash = hashlib.md5(img.tobytes()).hexdigest()

                if image_hash in hashes:
                    # If the hash is already in the list, remove the image
                    os.remove(os.path.join(folder_path, filename))
                else:
                    # Otherwise, add the hash to the list
                    hashes.append(image_hash)
                    
# Turns all images in a folder into grayscale
def convert_to_grayscale_recursive(folder_path):
    # Recursively traverse all subdirectories
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # Check if the file is a .jpg image
            if file.endswith('.jpg'):
                # Construct the full path to the file
                file_path = os.path.join(root, file)

                # Read the image
                img = cv2.imread(file_path)

                # Convert the image to grayscale
                gray_img = np.dot(img[...,:3], [0.299, 0.587, 0.114])

                # Save the grayscale image back to the same location
                cv2.imwrite(file_path, gray_img)

"""
    This function consolidates the 'train' and 'val' datasets from a given input folder into a single 
    dataset in the destination folder. The resulting dataset in the destination folder will not have 
    separate 'train' and 'val' directories, but just the class directories.
    
    Parameters:
    input_folder (str): The source directory where the 'train' and 'val' directories are located. 
        It assumes a structure like:
        input_folder/
            train/
                class1/
                class2/
                ...
            val/
                class1/
                class2/
                ...
    
    destination_folder (str): The destination directory where the merged data will be stored. 
        If the 'train' and 'val' directories and their class subdirectories do not exist, they will be created.
        
    The function goes through each 'train' and 'val' directory in the input folder, and for each class 
    it finds, it checks if a corresponding class directory exists in the destination folder. If not, it creates one.
    
    The function then copies the images from each class directory in the 'train' and 'val' directories 
    into the corresponding class directory in the destination folder. The images are renamed to avoid overwriting. 
    
    The new filenames are in the format 'frameX.jpg', where X is a number. The number for each new file 
    is determined by counting the existing files in the destination class directory and continuing from that count.
    
    This function assumes that the images in the source directory follow the naming convention 'frameX.jpg' 
    where X is a number. Images in the source directory are sorted by this number before copying to maintain order.
    """  
def combine_train_val_datasets(input_folder, destination_folder):
    dataset_types = ['train', 'val']

    for dataset_type in dataset_types:
        current_dataset_path = os.path.join(input_folder, dataset_type)

        # Loop over each class directory in the current dataset (train/val)
        for class_name in os.listdir(current_dataset_path):
            class_path = os.path.join(current_dataset_path, class_name)

            destination_class_path = os.path.join(destination_folder, class_name)

            # Create destination class directory if it doesn't exist
            if not os.path.exists(destination_class_path):
                os.makedirs(destination_class_path)

            # List existing files in the destination directory and count them
            existing_files = [f for f in os.listdir(destination_class_path) if f.endswith('.jpg')]
            existing_files.sort(key=lambda f: int(f.replace('frame','').replace('.jpg','')))  # Sort by frame number
            count = len(existing_files)

            # List the files in the current class directory and sort them
            files = [f for f in os.listdir(class_path) if f.endswith('.jpg')]
            files.sort(key=lambda f: int(f.replace('frame','').replace('.jpg','')))  # Sort by frame number

            # Copy and rename each file in the current class directory to the destination class directory
            for filename in files:
                source_file_path = os.path.join(class_path, filename)
                destination_file_path = os.path.join(destination_class_path, 'frame{}.jpg'.format(count))
                shutil.copyfile(source_file_path, destination_file_path)
                count += 1
                
                
rename_images('datasets/resize/normal')