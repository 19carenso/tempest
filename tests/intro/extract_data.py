import os
import re
import shutil
import yaml 
import subprocess


cwd = os.getcwd()
rel_settings_path = '/settings/settings.yaml'
joined_path = cwd + rel_settings_path
settings_path = os.path.normpath(joined_path)

with open(settings_path, 'r') as file:
    settings = yaml.safe_load(file)

# Define your source directory and target directory
DIR_DATA_IN = settings['DIR_DATA_IN']
TARGET_DIR = os.path.normpath(cwd+"/cache")

# Define a regular expression pattern to extract days from the file names
timestamp_pattern = r'_(\d{10})\.\w+\.2D\.nc'

# Specify the start and end days you want to extract
start_i_t = 345600  # Day 11
last_i_t = 357120   # Day 12

def create_extract(input_nc_file, output_nc_file, box = [[2304, 3840], [7936, 8704]]):
        
    lat_start = box[0][0]
    lat_end = box[0][1]
    lon_start = box[1][0]
    lon_end = box[1][1]

    # Construct the ncks command as a list of arguments
    ncks_command = [
        "ncks",
        "-d", f"lat,{lat_start},{lat_end}",
        "-d", f"lon,{lon_start},{lon_end}",
        input_nc_file,
        output_nc_file
    ]

    # Run the ncks command using subprocess
    subprocess.run(ncks_command, check=True)

# Traverse the source directory and find the matching files
for root, _, files in os.walk(DIR_DATA_IN):
    for filename in sorted(files):
        # Check if the file matches the regular expression pattern
        match = re.search(timestamp_pattern, filename)
        if match:
            day = int(match.group(1))
            print(day)
            # Check if the day falls within the desired range
            if start_i_t <= day <= last_i_t:
                var_check  = False
                for var_id in ['Precac', 'U10m', 'V10m'] :
                    if var_id in filename:
                        var_check = True 
                if var_check:
                    source_file = os.path.join(root, filename)
                    target_file = os.path.join(TARGET_DIR, filename)

                    # Create a symbolic link in the target directory
                    try:
                        create_extract(source_file, target_file)
                        print(f"Created symbolic link for {filename}")
                    except FileExistsError:
                        print(f"Symbolic link for {filename} already exists. Skipping.")
print("Extraction complete.")