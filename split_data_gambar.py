import os
import shutil
import random

def split_dataset(directory, train_ratio, val_ratio, test_ratio):
    # List all files in the directory
    all_files = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    
    # Shuffle the files
    random.shuffle(all_files)
    
    # Calculate the number of files for each split
    total_files = len(all_files)
    train_size = int(train_ratio * total_files)
    val_size = int(val_ratio * total_files)
    test_size = total_files - train_size - val_size
    
    # Split the files
    train_files = all_files[:train_size]
    val_files = all_files[train_size:train_size + val_size]
    test_files = all_files[train_size + val_size:]
    
    return train_files, val_files, test_files

def copy_files(files, destination):
    if not os.path.exists(destination):
        os.makedirs(destination)
    for file in files:
        shutil.copy(file, destination)

# Paths
data_dir = 'Batagor_dataset_no_duplicat'
train_dir = 'train'
val_dir = 'valid'
test_dir = 'test'

# Split the dataset
train_files, val_files, test_files = split_dataset(data_dir, 0.7, 0.15, 0.15)

# Copy files to respective directories
copy_files(train_files, train_dir)
copy_files(val_files, val_dir)
copy_files(test_files, test_dir)

print(f"Training set: {len(train_files)} images")
print(f"Validation set: {len(val_files)} images")
print(f"Testing set: {len(test_files)} images")
