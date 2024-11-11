import pandas as pd
import json
from datetime import datetime

# Define file paths
excel_file = r"D:\yil708\meta_data_all\MetaData\Code-ReID\VIGIL-ReID\scripts\Stoat_ReID.xlsx"
json_output_file = r"D:\yil708\meta_data_all\MetaData\Code-ReID\VIGIL-ReID\data\stoat.json"

# Load the Excel file
df = pd.read_excel(excel_file)

# Initialize the JSON structure
json_data = {
    "dataset_info": {
        "total_images": len(df),
        "num_classes": len(df["Class"].unique()) if "Class" in df.columns else "unknown",
        "metadata_features": ["temperature", "humidity", "rain", "camera_id", "angle"]
    },
    "images": []
}

# Convert each row in the DataFrame to the JSON format
for _, row in df.iterrows():
    # Process timestamp by combining Date and Time if both exist
    timestamp = None
    if pd.notna(row.get("Date")) and pd.notna(row.get("Time")):
        date_str = f"{row['Date']} {row['Time']}"
        try:
            # Attempt to parse with expected format
            timestamp = datetime.strptime(date_str, "%m/%d/%Y %H:%M:%S").strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            # Handle unexpected formats
            print(f"Warning: Date and Time format issue for row {row['FileName']}. Skipping timestamp.")
    
    # Append each image record with metadata
    image_record = {
        "img_path": row["FileName"],
        "class_label": int(row.get("Class", 0)),  # Default to 0 if Class is missing
        "class_name": row.get("Class Name", "Stoat"),
        "metadata": {
            "temperature": row.get("Temperature"),
            "humidity": row.get("Relative Humanity(%)"),
            "rain": row.get("Rain(mm)"),
            "camera_id": str(row.get("Camera_ID")).replace(",", "").strip(),
            "angle": row.get("Angle", "Unknown"),
            "timestamp": timestamp
        }
    }
    json_data["images"].append(image_record)

# Save JSON data to file
with open(json_output_file, "w") as json_file:
    json.dump(json_data, json_file, indent=4)

print(f"JSON file saved to {json_output_file}")
