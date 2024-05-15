import os
import hashlib
from PIL import Image

def calculate_hash(image_path):
    with Image.open(image_path) as img:
        hash_md5 = hashlib.md5(img.tobytes()).hexdigest()
    return hash_md5

def find_duplicate_images(directory):
    hashes = {}
    duplicates = []

    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('png', 'jpg', 'jpeg', 'gif', 'bmp')):
                file_path = os.path.join(root, file)
                try:
                    img_hash = calculate_hash(file_path)
                    if img_hash in hashes:
                        duplicates.append((file_path, hashes[img_hash]))
                    else:
                        hashes[img_hash] = file_path
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")

    return duplicates

# Usage
directory = 'Batagor_dataset_no_duplicat'
duplicates = find_duplicate_images(directory)

if duplicates:
    print(f"Total duplicate images found: {len(duplicates)}")
    print("Duplicate images:")
    for dup in duplicates:
        print(f"Duplicate: {dup[0]} is the same as {dup[1]}")
else:
    print("No duplicate images found.")
