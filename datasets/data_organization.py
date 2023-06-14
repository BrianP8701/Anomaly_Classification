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

source_dir = 'datasets/resize_datasets'
target_dir = 'datasets/resize'
val_size = 0.15  # 20% of the data will be in the validation set

train_val_split(source_dir, target_dir, val_size)
