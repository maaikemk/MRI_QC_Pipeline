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

### FLIRT & FNIRT of average brain to MNI brain

# # Register brain to MNI template by firstly applying linear registration
# average_T1 = 'average_T1_12MO'
# reference = 'infant_1yr_intensity_withSkull_1mm'
# affine_output = 'affine_output_12mo.mat'
# output_affine = 'out_flirt_12mo.nii.gz'
# myCommand = f'flirt -ref {reference} -in "{average_T1}" -omat "{affine_output}" -out "{output_affine}"'
# print('Running my Command:', myCommand)
# os.system(myCommand)
# print('My Command completed:', myCommand)

# # Register brain to MNI template by secondly applying non-linear b-spline registration
# reference2 = 'infant_1yr_intensity_withSkull_1mm'
# configuration = '--config=T1_2_MNI152_2mm.cnf'
# warp_output = 'cout_fnirt_12mo'
# warpfield_output = 'fout_fnirt_12mo'
# output_warped_structural = 'iout_fnirt_12mo.nii.gz'
# myCommand = f'fnirt --in="{average_T1}" --ref="{reference2}" --aff="{affine_output}" --cout="{warp_output}" --fout="{warpfield_output}" --iout="{output_warped_structural}" --subsamp=8,4,2,2'
# # myCommand = f'fnirt --in="{output_no_neck}" --ref="{reference2}" --aff="{affine_output}" --cout="{warp_output}" {configuration} --fout="{warpfield_output}" --iout="{output_warped_structural}"'
# print('Running my Command:', myCommand)
# os.system(myCommand)
# print('My Command completed:', myCommand)


### Test NN of infant atlas to average image

# ## Linear nearest neighbour alignment of AAL mask to registered image
# reference = 'average_T1_12MO_brain_extract.nii.gz'
# input_atlas = 'infant-1yr-intensity-1mm.nii'
# # input_atlas = 'infant-1yr-aal-1mm.nii'
# output_warped_atlas = 'infant_temp_to_average_brain.nii'
# affine_output = 'affine_output_12mo_temp_to_avg.mat'
# # input_atlas = 'AAL.nii'  # Assuming this file is in the current directory
# interpol_method = '-interp nearestneighbour'
# myCommand = f'flirt -ref "{reference}" -in "{input_atlas}" -out "{output_warped_atlas}" -omat "{affine_output}" {interpol_method}'
# # myCommand = f'flirt -ref "{reference}" -in "{input_atlas}" -out "{output_warped_atlas}" {interpol_method}'
# print('Running my Command:', myCommand)
# os.system(myCommand)
# print('My Command completed:', myCommand)

# ## Apply warp
# myCommand = f'flirt -in infant-1yr-aal-1mm.nii -ref average_T1_12MO_brain_extract.nii.gz -out infant_atlas_to_average_brain_with_initaff.nii -init affine_output_12mo_temp_to_avg.mat -applyxfm -interp nearestneighbour'
# print('Running my Command:', myCommand)
# os.system(myCommand)
# print('My Command completed:', myCommand)


def get_unique_identifiers(folder_path):
    # Set to store unique identifiers
    unique_identifiers = set()
    
    # Iterate through each file in the folder
    for filename in os.listdir(folder_path):
        # Ensure we are only processing files (ignore directories)
        if os.path.isfile(os.path.join(folder_path, filename)):
            # Split the filename to get the identifier
            identifier = filename.split('_')[0]
            # Add the identifier to the set
            unique_identifiers.add(identifier)
    
    return unique_identifiers

def write_identifiers_to_file(identifiers, output_file):
    # Write each identifier to a new line in the output file
    with open(output_file, 'w') as file:
        for identifier in sorted(identifiers):
            file.write(f"{identifier}\n")

# Example usage:
folder_path = r'C:\Users\maaik\OneDrive\Documents\UU\MinorResearchProject\Stefan\shellCommandTest\RealWrapperData\Lowfield_fMRItest\MetaOutputsGUI_NN_RFset'
output_file = r'C:\Users\maaik\OneDrive\Documents\UU\MinorResearchProject\Stefan\shellCommandTest\RealWrapperData\Lowfield_fMRItest\identifiers.txt'

unique_identifiers = get_unique_identifiers(folder_path)
write_identifiers_to_file(unique_identifiers, output_file)

print(f"Unique identifiers have been written to {output_file}")