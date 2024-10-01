import os
import tkinter as tk
from tkinter import filedialog
import shutil
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import math
from nireports.reportlets.nuisance import plot_qi2
from scipy.stats import chi2
from sklearn.neighbors import KernelDensity
import os.path as op
import PIL
from PIL import Image
import pandas as pd
import os  
import sys
import shutil

###PREPROCESSING
# # Create a Tkinter root window
# root = tk.Tk()
# root.withdraw()  

# # Ask the user to select the input folder
# input_folder = filedialog.askdirectory(title="Select Input Folder")

# # Specify the folder containing output files
# output_folder = 'AverageBrainTest_noBET'

# # Check if the output folder exists, if not, create it
# if not os.path.exists(output_folder):
#     os.makedirs(output_folder)
#     print(f"Output folder '{output_folder}' created.")

# if input_folder:
#     # Get a list of files in the input folder
#     input_files = os.listdir(input_folder)
#     nifti_files = [f for f in input_files if 'T1' in f and f.endswith('.nii')]
#     # Loop over each input file
#     for file in nifti_files:
#         # Construct full input and output paths for each step in the pipeline for NIfTI input files 
#         if file.endswith('.nii') or file.endswith('.nii.gz'):           
#             input_path = os.path.join(input_folder, file)
#             output_prefix_tmp = os.path.splitext(file)[0]  
#             output_prefix = os.path.splitext(output_prefix_tmp)[0]  # Extract filename without extension
#             output_reor = os.path.join(output_folder, output_prefix + '_reor.nii.gz')
#             output_no_neck = os.path.join(output_folder, output_prefix + '_noNeck.nii.gz')
#             output_extracted = os.path.join(output_folder, output_prefix + '_extracted.nii.gz')
#             output_norm = os.path.join(output_folder, output_prefix + '_norm.nii.gz')

#             # Reorient input image to MNI orientation
#             if not os.path.exists(output_reor):
#                 myCommand = f'fslreorient2std "{input_path}" "{output_reor}"'
#                 print('Running my Command:', myCommand)
#                 os.system(myCommand)
#                 print('My Command completed:', myCommand)

#             # Neck removal
#             if not os.path.exists(output_no_neck):
#                 myCommand = f'robustfov -i "{output_reor}" -r "{output_no_neck}"'
#                 print('Running my Command:', myCommand)
#                 os.system(myCommand)
#                 print('My Command completed:', myCommand)

#             # # Brain extraction
#             # if not os.path.exists(output_extracted):
#             #     myCommand = f'bet "{output_no_neck}" "{output_extracted}" -A'
#             #     print('Running my Command:', myCommand)
#             #     os.system(myCommand)
#             #     print('My Command completed:', myCommand)

#             # Normalize image intensity 
#             ###step is skipped now
#             if not os.path.exists(output_norm):
#                 myCommand = f'mri_normalize "{output_no_neck}" "{output_norm}"'
#                 # myCommand = f'mri_normalize "{output_extracted}" "{output_norm}"'
#                 print('Running my Command:', myCommand)
#                 os.system(myCommand)
#                 print('My Command completed:', myCommand)

def load_nifti_files(folder_path):
    # Get a list of all files in the folder
    all_files = os.listdir(folder_path)
    # Filter files that contain 'T1' in their name and have a '.nii' or '.nii.gz' extension
    nifti_files = [f for f in all_files if f.endswith('_affine.nii.gz')]
    # nifti_files = [f for f in all_files if 'T1' in f]
    return nifti_files

def load_nifti_data(file_path):
    # Load a NIfTI file and return the image data as a numpy array
    img = nib.load(file_path)
    return img.get_fdata()

def main(folder_path, output_path):
    # Load all NIfTI files that contain 'T1' in their name
    nifti_files = load_nifti_files(folder_path)

    if not nifti_files:
        print("No NIfTI files with 'T1' found in the specified folder.")
        return

    # Initialize a list to store the data arrays
    data_arrays = []

    # Load each NIfTI file and add its data to the list
    for nifti_file in nifti_files:
        file_path = os.path.join(folder_path, nifti_file)
        data = load_nifti_data(file_path)
        print(data.shape)
        data_arrays.append(data[:,:,:])

    # Calculate the average across all the loaded data arrays
    average_data = np.mean(data_arrays, axis=0)

    min_val = np.min(average_data)
    max_val = np.max(average_data)

    # Perform min-max normalization
    normalized_data = (average_data - min_val) / (max_val - min_val)

    # Save the average data as a new NIfTI file
    # Use the affine of the first image for the new NIfTI file
    first_img = nib.load(os.path.join(folder_path, nifti_files[0]))
    average_img = nib.Nifti1Image(normalized_data, affine=first_img.affine)

    # Save the resulting NIfTI file
    nib.save(average_img, output_path)
    print(f"Average NIfTI file saved to {output_path}")

output_folder = "C:/Users/maaik/OneDrive/Documents/UU/MinorResearchProject/Stefan/shellCommandTest/RealWrapperData/Lowfield_fMRItest/lf_12MO_ToRandomOne"
output_path = "C:/Users/maaik/OneDrive/Documents/UU/MinorResearchProject/Stefan/shellCommandTest/RealWrapperData/Lowfield_fMRItest/average_T1_12MO.nii.gz"  # Replace with your output file path

# output_path = "/c/mnt/Users/maaik/OneDrive/Documents/UU/MinorResearchProject/visual_code_stud/project-1/LEAP24mo_avg/average_T1_v2.nii.gz"  # Replace with your output file path
main(output_folder, output_path)

# else:
# print("No folder selected. Exiting...")

