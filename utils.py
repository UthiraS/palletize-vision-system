#!/usr/bin/python3
from PIL import Image
import os, sys
import cv2
import glob

#function to rename the files in data and number them
def rename():

    data_path = "data/Pallets/*"  # Added wildcard to match files
    
    # Enumerate files to get numbers for renaming
    for index, file in enumerate(glob.glob(data_path), start=1):
        # Get file extension
        file_extension = os.path.splitext(file)[1]
        
        # Create new filename with number
        new_name = f"data/Pallets/pallet_{index}{file_extension}"
        
        # Rename the file
        os.rename(file, new_name)
        print(f"Renamed {file} to {new_name}")



if __name__ == "__main__":
    rename()
