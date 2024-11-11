import pandas as pd
import json
from datetime import datetime

# Define file paths
txt_file = r"D:\yil708\meta_data_all\MetaData\Code-ReID\VIGIL-ReID\scripts\stoatjson.txt"
json_output_file = r"D:\yil708\meta_data_all\MetaData\Code-ReID\VIGIL-ReID\scripts\stoat.json"

# Load the text file into a DataFrame, assuming tab separation
df = pd.read_csv(txt_file, sep='\t')

# Initialize the JSON structure
json_data = {
    "dataset_info": {
        "total_images": len(df),
        "num_classes": len(df["Camera_ID"].unique()),  # Assuming each Camera_ID represents a class here
        "metadata_features": ["temperature", "humidity", "rain", "camera_id", "angle"]
    },
    "images": []
}

# Convert each row in the DataFrame to the JSON format
for _, row in df.iterrows():
    # Combine 'Date' and 'Time' into a timestamp in the desired format
    date_str = f"{row['Date']} {row['Time']}"
    try:
        timestamp = datetime.strptime(date_str, "%m/%d/%Y %H:%M:%S").strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        timestamp = None  # If parsing fails, leave timestamp as None

    # Create an image record based on the row data
    image_record = {
        "img_path": row["FileName"],
        "class_label": int(row.get("Class", 0)),  # Assuming a default class if not provided
        "class_name": "Stoat",  # Assuming the class name for all is "Stoat" as per your previous data
        "metadata": {
            "temperature": row["Temperature"],
            "humidity": row["Relative Humanity(%)"],
            "rain": row["Rain(mm)"],
            "camera_id": str(row["Camera_ID"]).strip(),
            "angle": row["Angle"],
            "timestamp": timestamp
        }
    }
    json_data["images"].append(image_record)

# Save JSON data to file
with open(json_output_file, "w") as json_file:
    json.dump(json_data, json_file, indent=4)

print(f"JSON file saved to {json_output_file}")
