import os
import shutil
from sklearn.model_selection import train_test_split

def train_val_split(source_dir, target_dir, val_size):
    # Create train and val directories
    os.makedirs(os.path.join(target_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(target_dir, 'val'), exist_ok=True)

    # For each class in the dataset
    for class_dir in os.listdir(source_dir):
        if os.path.isdir(os.path.join(source_dir, class_dir)):
            # Get all the images in this class
            images = os.listdir(os.path.join(source_dir, class_dir))
            
            # Split images into train and validation sets
            train_images, val_images = train_test_split(images, test_size=val_size, random_state=42)

            # Create target directories for this class
            os.makedirs(os.path.join(target_dir, 'train', class_dir), exist_ok=True)
            os.makedirs(os.path.join(target_dir, 'val', class_dir), exist_ok=True)

            # Move images to their respective sets
            frame = 0
            for img in train_images:
                shutil.move(os.path.join(source_dir, class_dir, img), os.path.join(target_dir, 'train', class_dir, f'frame{frame}.jpg'))
                frame += 1
            frame = 0
            for img in val_images:
                shutil.move(os.path.join(source_dir, class_dir, img), os.path.join(target_dir, 'val', class_dir, f'frame{frame}.jpg'))
                frame += 1
                
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


source_dir = 'datasets/resize_datasets'
target_dir = 'datasets/resize'
val_size = 0.15  # 20% of the data will be in the validation set

train_val_split(source_dir, target_dir, val_size)
