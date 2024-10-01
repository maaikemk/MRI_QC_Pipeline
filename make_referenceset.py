import tkinter as tk
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import os
from collections import defaultdict
from tkinter import filedialog  # Import file dialog from tkinter
import shutil
import matplotlib.colors
import glob
import re

# Define the directory path of the images and of the txt files (initially set to None)
folder_paths = []

def select_folder_paths():
    """
    Select multiple folder paths using a file dialog and store them in the global variable folder_paths.

    Returns:
        None
    """
    global folder_paths
    while True:
        folder_path = filedialog.askdirectory()  # Open file dialog for selecting a folder
        if folder_path:
            folder_paths.append(folder_path)
            print("Selected folder path:", folder_path)
        else:
            break

def analyse_images_and_close_window():
    """
    Closes the Tkinter first window.

    Returns:
        None
    """
    # Close the window/
    root.destroy()

# Create a root window
root = tk.Tk()

# Create buttons to open file dialogs for selecting folders
select_folder_button = tk.Button(root, text="Select Images Folders", command=select_folder_paths)
select_folder_button.pack()

# Create a button to analyze selected images and close the window
analyse_button = tk.Button(root, text="Analyse Selected Images", command=analyse_images_and_close_window)
analyse_button.pack()

# Run the root window
root.mainloop()

# Create a new folder to combine all files
combined_folder = "Reference_Set"
if os.path.exists(combined_folder):
    folder_exists = True
else:
    folder_exists = False
    os.makedirs(combined_folder, exist_ok=True)

    # Copy all files from selected folders to the new combined folder
    for folder_path in folder_paths:
        for filename in os.listdir(folder_path):
            shutil.copy(os.path.join(folder_path, filename), os.path.join(combined_folder, filename))

# Now use the combined_folder in the rest of the code
folder_path = combined_folder  
print(folder_path)

# Create a defaultdict to store files by identifier
files_by_identifier = defaultdict(list)
files_txt = []
snrlist_txt = []
snrDlist_txt = []
brainmask_nii = []

# Define the order of tags and corresponding index
tag_order = {"signal.nii": 0, "struct.nii.gz": 1,  "std.nii": 2, "snr.nii": 3, "snrD.nii": 4, "field.nii": 5, "fieldZmap.nii": 6, "snrZmap.nii.gz": 7, "snrDZmap.nii.gz": 8}  # Example order and indices

# List all files in the directory
files = os.listdir(folder_path)

# Iterate through each file
for file_name in files:
    # print(file_name)
    # Split the file name into identifier and tag
    identifier, tag = file_name.split("_", 1)
    if tag == 'qm.txt':
        files_txt.append(folder_path + '/'+ file_name)
    elif tag == 'snrZmap.txt':
        snrlist_txt.append(file_name)
    elif tag == 'snrDZmap.txt':
        snrDlist_txt.append(file_name)
    elif tag == 'brainmask.nii':
        brainmask_nii.append(file_name)
    else:
        # Add the file to the corresponding identifier list
        files_by_identifier[identifier].append((file_name, tag_order.get(tag, len(tag_order))))  # Store file and index

# Sort files within each identifier group based on indices
for identifier, file_list in files_by_identifier.items():
    sorted_files = sorted(file_list, key=lambda x: x[1])  # Sort based on index
    files_by_identifier[identifier] = sorted_files

# Initialize a list to store the grouped and sorted files
grouped_sorted_lists = []

# Populate the grouped_sorted_lists with sorted files
for identifier, sorted_files in files_by_identifier.items():
    file_list = [file_name for file_name, _ in sorted_files]  # Extract file names
    grouped_sorted_lists.append(file_list)

nifti_data = grouped_sorted_lists
print('Made lists')

#############

def calculate_Zmap(list_txt):  
    """
    Make a list of list of all values per brain region of all txt files. Calculate 
    the average and standard deviation per brain region.

    list_txt: list of str, list of filenames containing data in the format of 'brain 
    region name: brain region value \n'

    Returns tuple of 
    avg_reg: 1D numpy array, containing average values per brain region
    std_reg: 1D numpy array, containing standard deviations per brain region
    region_arrays: list of lists, arrays containing region data of all files in the dataset
    """
    region_arrays = [[]]
    region_values = []
    for i, filename in enumerate(list_txt):  
        with open(folder_path + '/'+ filename, 'r') as file:
            lines = file.readlines()                
            for j, line in enumerate(lines):
                entries = line.strip().split()
                if len(entries) >= 2:
                    if len(region_arrays) < i+1:
                        region_arrays.append([])
                    value = float(entries[1])
                    region_arrays[i].append(value)
                    if i == 0: 
                        # Collect brain region labels one time
                        region_values.append(entries[0])
    region_arrays = np.array(region_arrays, dtype=np.float64)                
    region_arrays[np.isinf(region_arrays)] = np.nan
    avg_reg = np.nanmean(region_arrays, axis=0)
    std_reg = np.nanstd(region_arrays, axis=0)

    return avg_reg, std_reg, region_arrays, region_values

def calculate_z_def_fields(output_filename):
    """
    Calculate the average and standard deviation per voxel of the deformation fields NIfTI files
    which can be used for z-field calculations.

    Returns:
    None
    """
    warpfields = []

    # List all files in the directory
    all_files = os.listdir(combined_folder)

    # Filter files ending with '_fieldZmap.nii'
    nifti_files = [f for f in all_files if f.endswith('_fieldZmap.nii')]

    # Load each NIfTI file and add the data to the list
    for nifti_file in nifti_files:
        file_path = os.path.join(combined_folder, nifti_file)
        nifti_img = nib.load(file_path)
        nifti_data = nifti_img.get_fdata()
        warpfields.append(nifti_data)

    avg_warpfield = np.mean(warpfields, axis=0)
    std_warpfield = np.std(warpfields, axis=0)

    with open(combined_folder + '/' + output_filename,'w') as f:
        # Write the first 3D matrix
        for i, matrix_2d in enumerate(avg_warpfield):
            f.write(f'# Matrix 1 - Slice {i}\n')  # Add a comment to indicate the slice index
            np.savetxt(f, matrix_2d, fmt='%.6f')
            f.write('\n')  # Add a newline to separate matrices

        # Write the second 3D matrix
        for i, matrix_2d in enumerate(std_warpfield):
            f.write(f'# Matrix 2 - Slice {i}\n')  # Add a comment to indicate the slice index
            np.savetxt(f, matrix_2d, fmt='%.6f')
            f.write('\n')  # Add a newline to separate matrices
 
def save_z_maps(avg_list,std_list,region_values,file_name):
    with open(combined_folder + '/' + file_name,'w') as Zmap:
        for idx in range(len(region_values)):
            Zmap.write(str(avg_list[idx]) + ' ' + str(std_list[idx]) + ' ' + str(region_values[idx]) + '\n')

def remove_identical_struct_nifti(directory):
    tag = "_struct.nii.gz"
    nifti_files = glob.glob(os.path.join(directory, f"*{tag}"))
    identifiers_to_remove = set()

    def read_nifti_image(file_path):
        img = nib.load(file_path)
        return img.get_fdata()

    def compare_images(image1, image2):
        return np.array_equal(image1, image2)

    def extract_identifier(file_name, tag):
        return file_name.split(tag)[0]

    def remove_files_with_identifier(identifier):
        files_to_remove = glob.glob(os.path.join(directory, f"{identifier}*"))
        for file_path in files_to_remove:
            os.remove(file_path)
            print(f"Removed: {file_path}")

    # Compare all NIfTI files to find identical ones
    for i in range(len(nifti_files)):
        if nifti_files[i] in identifiers_to_remove:
            continue
        img1 = read_nifti_image(nifti_files[i])
        identifier_i = extract_identifier(os.path.basename(nifti_files[i]), tag)
        for j in range(i + 1, len(nifti_files)):
            if nifti_files[j] in identifiers_to_remove:
                continue
            img2 = read_nifti_image(nifti_files[j])
            if compare_images(img1, img2):
                identifier_j = extract_identifier(os.path.basename(nifti_files[j]), tag)
                identifiers_to_remove.add(nifti_files[j])
                remove_files_with_identifier(identifier_j)

    # Remove the identified files
    for file_to_remove in identifiers_to_remove:
        os.remove(file_to_remove)
        print(f"Removed: {file_to_remove}")       

##################

remove_identical_struct_nifti(combined_folder)
# files_txt_ref, snrlist_txt_ref, snrDlist_txt_ref = qm_files_reference(combined_folder)
avg_reg, std_reg, region_arrays, region_values = calculate_Zmap(snrlist_txt)
avg_reg2, std_reg2, region_arrays2, region_values2 = calculate_Zmap(snrDlist_txt)
save_z_maps(avg_reg, std_reg, region_values,'total_zmap_SNR.txt')
save_z_maps(avg_reg2, std_reg2, region_values2,'total_zmap_SNRD.txt')
calculate_z_def_fields('total_zmap_fields.txt')
