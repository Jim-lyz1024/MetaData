import os
from PIL import Image
from PIL.ExifTags import TAGS
import pandas as pd
from datetime import datetime

# Define the directory with the images
image_dir = '/raid/yil708/stoat_data/auxiliary_network_pics/labelled_auxiliary_network_pics/labelled_auxiliary_network_pics/'

# List to store the extracted data
image_data = []

# Function to extract capture time from image EXIF data
def extract_capture_time(image_path):
    try:
        img = Image.open(image_path)
        exif_data = img._getexif()
        if exif_data:
            for tag, value in exif_data.items():
                if TAGS.get(tag) == 'DateTimeOriginal':
                    # Convert the DateTimeOriginal to MM/DD/YYYY and 24-hour time format
                    capture_time = datetime.strptime(value, '%Y:%m:%d %H:%M:%S')
                    date = capture_time.strftime('%m/%d/%Y')
                    time = capture_time.strftime('%H:%M')
                    return date, time
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
    return "/", "/"  # Return / for missing Date and Time

# Iterate through all images in the directory
for image_name in os.listdir(image_dir):
    if image_name.lower().endswith(('jpg', 'jpeg', 'png', 'tiff')):
        image_path = os.path.join(image_dir, image_name)
        date, time = extract_capture_time(image_path)
        image_data.append([image_name, date, time])

# Convert the data into a pandas DataFrame
df = pd.DataFrame(image_data, columns=['Image Name', 'Date', 'Time'])

# Save the DataFrame to an Excel file
output_excel_path = 'image_capture_times_with_missing.xlsx'
df.to_excel(output_excel_path, index=False)

print(f"Image capture times (including missing) saved to {output_excel_path}")
