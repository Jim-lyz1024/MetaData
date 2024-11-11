import os

# Define the base directory path
base_dir = r"D:\yil708\meta_data_all\MetaData\Code-ReID\VIGIL-ReID\data\Stoat"

# Create an empty list to store the Camera_IDs
camera_ids = []

# Traverse all folders and files in the specified directory
for root, dirs, files in os.walk(base_dir):
    for file in files:
        # Check if the file is an image (adjust extensions as needed)
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            # Split the filename by underscores and extract the Camera_ID (second part)
            camera_id = file.split('_')[1]
            # Append the Camera_ID to the list
            camera_ids.append(camera_id)

# Print the list of Camera_IDs
for cam_id in camera_ids:
    print(cam_id)

# If you need to save the results to a text file, use the following code
with open("camera_ids.txt", "w") as f:
    for cam_id in camera_ids:
        f.write(cam_id + "\n")
