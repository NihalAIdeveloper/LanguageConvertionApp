import splitfolders
import os

# Define the source directory and the target directory
source_dir = 'SignImage48x48'
target_dir = 'splitdataset48x48'

# Ensure the target directory does not already exist to avoid overwriting
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# Split the dataset into training and validation sets
splitfolders.ratio(source_dir, target_dir, ratio=(0.8, 0.2))

print(f"Dataset has been split and saved to {target_dir}")
