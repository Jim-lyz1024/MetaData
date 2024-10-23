import os

# Define file paths for the directory containing images and the text file with image names
folder_path = r"E:\LYZ\AucklandCourse\2024Thesis\Metadata\stoat\labelled_auxiliary_network_pics"
image_names_file = r"E:\LYZ\AucklandCourse\2024Thesis\Metadata\stoat\labelled_stoat.txt"

# Read image names from the text file (first item in each line is the filename)
with open(image_names_file, 'r') as f:
    image_names_in_txt = [line.split()[0] for line in f.readlines()]

# Get all image files from the directory
actual_image_files = set(os.listdir(folder_path))

# Find images that are in the text file but missing from the directory
missing_in_folder = [image for image in image_names_in_txt if image not in actual_image_files]

# Find images that are in the directory but not mentioned in the text file
missing_in_txt = [image for image in actual_image_files if image not in image_names_in_txt]

# Print results
if missing_in_folder:
    print("In image_names.txt, but not in folder:")
    for image in missing_in_folder:
        print(image)
else:
    print("Images in image_names.txt all exist in folder")

if missing_in_txt:
    print("\nIn folder, but not in image_names.txt:")
    for image in missing_in_txt:
        print(image)
else:
    print("\nImages in folder all exist in image_names.txt")