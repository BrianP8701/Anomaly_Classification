'''
    This file assumes you have your images split into classes in the following format:
    
    folder/
        class1/
            frame0.jpg
            frame1.jpg
            ...
        class2/
        ...
    
    If not, you can extract images from a video in the extract_images_from_video.py file.
    And crop images in the crop_images.py file.
    
    The prepare_data method will streamline the preprocessing step.
    
    At the end of any of these functions, you will have a folder with processed, augmented images split into train, val, and test folders, ready for training.
'''
import data_processing.preprocessing_methods as preprocessing
import data_processing.data_organization_methods as data_organization_methods
import shutil
import os
from PIL import Image
import re

# This method assumes you haven't split your dataset into train, val, and test folders yet.
def prepare_data_and_split(dataset_path, destination_path, augmentations_per_class=0, flips_per_class=0, gmms=False, num_components=6, grayscale=False, resize=224, train_val_test_split=[0.7, 0.2, 0.1]):
    # Copy dataset to destination path
    shutil.copytree(dataset_path, destination_path)

    # Loop through each class folder
    for subfolder in os.listdir(destination_path):
        subfolder_path = os.path.join(destination_path, subfolder)
        
        if subfolder == ".DS_Store":  # Check for .DS_Store and skip if found
            continue
        
        # Resize
        for img in os.listdir(subfolder_path):
            if not re.match(r'.*\.(jpg|jpeg)$', img, re.IGNORECASE):  # Check if it's a .jpg or .jpeg file
                continue
            img_path = os.path.join(subfolder_path, img)
            preprocessing.resize_image(img_path, img_path, resize)
            
        # Grayscale
        if grayscale:
            preprocessing.convert_to_grayscale_recursive(subfolder_path)
        
        # Apply GMMS preprocessing
        if gmms:
            for img in os.listdir(subfolder_path):
                if not re.match(r'.*\.(jpg|jpeg)$', img, re.IGNORECASE):  # Check if it's a .jpg or .jpeg file
                    continue
                img_path = os.path.join(subfolder_path, img)
                new_img = Image.fromarray(preprocessing.gmms_preprocess_image(img_path, 6))
                new_img.save(img_path)
        
    # Data augmentation
    if augmentations_per_class > 0:
        preprocessing.augment_dataset_2(destination_path, augmentations_per_class)
    if flips_per_class > 0:
        preprocessing.flip_images_in_directory(destination_path, flips_per_class, 1)
        
    # Split dataset into train, val, and test folders
    data_organization_methods.split_dataset_in_place(destination_path, *train_val_test_split)
    
    # Delete empty class folders
    for subfolder in os.listdir(destination_path):
        subfolder_path = os.path.join(destination_path, subfolder)
        
        if os.path.isdir(subfolder_path) and subfolder not in ['train', 'val', 'test']:
            shutil.rmtree(subfolder_path)
        elif os.path.isdir(subfolder_path):
            # Reorder images
            data_organization_methods.reorder_images(subfolder_path)
            


# This method assumes you have already split your dataset into train, val, and test folders.
def prepare_data(dataset_path, destination_path, augmentations_per_class=0, flips_per_class=0, gmms=False, num_components=6, grayscale=False, resize=224):
    # Copy dataset to destination path
    shutil.copytree(dataset_path, destination_path)

    # Loop through train, val, and test folders
    for phase in os.listdir(destination_path):
        phase_path = os.path.join(destination_path, phase)

        # Loop through each class folder
        for subfolder in os.listdir(phase_path):
            subfolder_path = os.path.join(phase_path, subfolder)
            
            # Resize
            for img in os.listdir(subfolder_path):
                if not re.match(r'.*\.(jpg|jpeg)$', img, re.IGNORECASE):  # Check if it's a .jpg or .jpeg file
                    continue
                img_path = os.path.join(subfolder_path, img)
                preprocessing.resize_image(img_path, img_path, resize)
                
            # Grayscale
            if grayscale:
                preprocessing.convert_to_grayscale_recursive(subfolder_path)
            
            # Apply GMMS preprocessing
            if gmms:
                for img in os.listdir(subfolder_path):
                    if not re.match(r'.*\.(jpg|jpeg)$', img, re.IGNORECASE):  # Check if it's a .jpg or .jpeg file
                        continue
                    img_path = os.path.join(subfolder_path, img)
                    new_img = Image.fromarray(preprocessing.gmms_preprocess_image(img_path, 6))
                    new_img.save(img_path)
    
        # Data augmentation
        if augmentations_per_class > 0:
            preprocessing.augment_dataset_2(phase_path, augmentations_per_class)
        if flips_per_class > 0:
            preprocessing.flip_images_in_directory(phase_path, flips_per_class)