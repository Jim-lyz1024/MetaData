import pandas as pd

# Define the file paths
file_path = r"D:\yil708\meta_data_all\MetaData\Code-ReID\VIGIL-ReID\scripts\Stoat_ReID.xlsx"
sorted_file_path = r"D:\yil708\meta_data_all\MetaData\Code-ReID\VIGIL-ReID\scripts\sorted_Stoat_ReID.xlsx"

# Load the Excel file
df = pd.read_excel(file_path)

# Split the 'FileName' column by underscore and create new columns for sorting
df[['Prefix', 'ID', 'CameraID', 'Count']] = df['FileName'].str.extract(r'(\w+\\)?(\d+)_(\w+)_(\d+)', expand=True)

# Convert 'ID' and 'Count' to integers for numerical sorting; leave 'CameraID' as a string to handle alphanumeric values
df['ID'] = pd.to_numeric(df['ID'], errors='coerce')
df['Count'] = pd.to_numeric(df['Count'], errors='coerce')

# Sort within each directory (e.g., "gallery", "train", etc.) by ID, then CameraID, and finally by Count
df_sorted = df.sort_values(by=['Prefix', 'ID', 'CameraID', 'Count']).reset_index(drop=True)

# Drop the helper columns after sorting
df_sorted = df_sorted.drop(columns=['ID', 'CameraID', 'Count'])

# Save the sorted DataFrame back to an Excel file
df_sorted.to_excel(sorted_file_path, index=False)

print("Sorted file saved at:", sorted_file_path)
