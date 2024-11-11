import os
import re

# Define the base directory path
base_dir = r"D:\yil708\meta_data_all\MetaData\Code-ReID\VIGIL-ReID\data\Stoat"

# Create a dictionary to store image paths grouped by subdirectories
image_dict = {}

# Traverse all folders and files in the specified directory
for root, dirs, files in os.walk(base_dir):
    parent_folder = os.path.basename(root)
    image_list = []
    
    for file in files:
        # Check if the file is an image (adjust extensions as needed)
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            # Extract ID, CameraID, and Count from filename using regex
            match = re.match(r"(\d+)_([A-Za-z0-9]+)_(\d+)", file)
            if match:
                # Get ID and Count as integers for sorting
                id_val = int(match.group(1))
                camera_id = match.group(2)  # CameraID may be alphanumeric
                count = int(match.group(3))
                # Append as a tuple (parent_folder, filename, id_val, camera_id, count) for sorting
                image_list.append((parent_folder, file, id_val, camera_id, count))
    
    # Sort images by ID, then by alphanumeric CameraID, and then by Count
    image_list = sorted(image_list, key=lambda x: (x[2], x[3], x[4]))
    image_dict[parent_folder] = image_list  # Store sorted list in dictionary

# Write the sorted filenames to a file
with open("sorted_image_names_with_parent.txt", "w") as f:
    for folder, images in image_dict.items():
        for image_info in images:
            # Format as "parent_folder\filename" for each sorted entry
            formatted_name = f"{image_info[0]}\\{image_info[1]}"
            print(formatted_name)
            f.write(formatted_name + "\n")
