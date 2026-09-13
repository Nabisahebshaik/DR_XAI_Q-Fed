import os
import zipfile

# Use r'' for raw string to handle Windows backslashes properly, 
# and remove the inner quotation marks.
zip_path = r"C:\Users\shaik\OneDrive\Desktop\sih_dr_project\archive (5).zip"
extract_dir = 'dataset'

if not os.path.exists(extract_dir):
    print("Extracting dataset... Please wait.")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(extract_dir)
    print("Dataset extracted successfully!")
else:
    print("Dataset folder already exists.")