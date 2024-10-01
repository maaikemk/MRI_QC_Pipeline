import os
import tkinter as tk
from tkinter import filedialog
import shutil
import nibabel as nib
import numpy as np
# import matplotlib.pyplot as plt
import math
from nireports.reportlets.nuisance import plot_qi2
from scipy.stats import chi2
from sklearn.neighbors import KernelDensity
import os.path as op
# import PIL
# from PIL import Image
# import pandas as pd
import os  
import sys
import shutil

###### Rigid registration of all low field scans to the average brain


###PREPROCESSING
# Create a Tkinter root window
root = tk.Tk()
root.withdraw()  

# Ask the user to select the input folder
input_folder = filedialog.askdirectory(title="Select Input Folder")

# Specify the folder containing output files
output_folder = 'Skull_affines_norm'

# Check if the output folder exists, if not, create it
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"Output folder '{output_folder}' created.")

if input_folder:
    # Get a list of files in the input folder
    input_files = os.listdir(input_folder)

    # Loop over each input file
    for file in input_files:
        # Construct full input and output paths for each step in the pipeline for NIfTI input files 
        if file.endswith('.nii') or file.endswith('.nii.gz'):           
            input_path = os.path.join(input_folder, file)
            output_prefix_tmp = os.path.splitext(file)[0]  
            output_prefix = os.path.splitext(output_prefix_tmp)[0]  # Extract filename without extension
            output_reor = os.path.join(output_folder, output_prefix + '_reor.nii.gz')
            output_no_neck = os.path.join(output_folder, output_prefix + '_noNeck.nii.gz')
            output_extracted = os.path.join(output_folder, output_prefix + '_extracted.nii.gz')
            output_norm =  os.path.join(output_folder, output_prefix + '_affineNorm.nii.gz')
            output_warped_structural = os.path.join(output_folder, output_prefix + '_struct.nii.gz')
            output_warped_atlas = os.path.join(output_folder, output_prefix + '_warpedAtlas.nii.gz')
            output_warped_structural_seg = os.path.join(output_folder, output_prefix + '_structSeg.nii.gz')
            warp_output = os.path.join(output_folder, output_prefix + '_nonlineartransf.nii.gz')
            warpfield_output = os.path.join(output_folder, output_prefix + '_warpfield.nii.gz')
            affine_output = os.path.join(output_folder, output_prefix + '_affineTransf_v2.mat')
            output_file_name = output_prefix + '_orig.nii.gz'
            final_output = os.path.join(output_folder, output_prefix + '_final.nii.gz')

            # # Reorient input image to MNI orientation
            # if not os.path.exists(output_reor):
            #     myCommand = f'fslreorient2std "{input_path}" "{output_reor}"'
            #     print('Running my Command:', myCommand)
            #     os.system(myCommand)
            #     print('My Command completed:', myCommand)

            # # Neck removal
            # if not os.path.exists(output_no_neck):
            #     myCommand = f'robustfov -i "{output_reor}" -r "{output_no_neck}"'
            #     print('Running my Command:', myCommand)
            #     os.system(myCommand)
            #     print('My Command completed:', myCommand)

            # Normalize image intensity 
            ###step is skipped now
            # if not os.path.exists(output_norm):
            #     myCommand = f'mri_normalize "{output_extracted}" "{output_norm}"'
            #     print('Running my Command:', myCommand)
            #     os.system(myCommand)
            #     print('My Command completed:', myCommand)

            # # Register brain to MNI template by firstly applying linear registration
            # if not os.path.exists(affine_output):
            #     reference = 'average_T1_afterFLIRTnorm'
            #     # reference = 'MNI152_T1_2mm_brain'
            #     myCommand = f'flirt -ref {reference} -in "{output_norm}" -omat "{affine_output}"'
            #     print('Running my Command:', myCommand)
            #     os.system(myCommand)
            #     print('My Command completed:', myCommand)

            # # Register brain to MNI template by secondly applying non-linear b-spline registration
            if not os.path.exists(warpfield_output):
                reference2 = 'average_T1_afterFLIRTnorm'
                # reference2 = 'MNI152_T1_2mm_brain'
                configuration = '--config=T1_2_MNI152_2mm'
                myCommand = f'fnirt --in="{output_no_neck}" --ref="{reference2}" --aff="{affine_output}" --cout="{warp_output}" --fout="{warpfield_output}" --iout="{output_warped_structural}"'
                # myCommand = f'fnirt --in="{output_no_neck}" --ref="{reference2}" --aff="{affine_output}" --cout="{warp_output}" {configuration} --fout="{warpfield_output}" --iout="{output_warped_structural}"'
                print('Running my Command:', myCommand)
                os.system(myCommand)
                print('My Command completed:', myCommand)

            # # Linear nearest neighbour alignment of AAL mask to registered image
            # if not os.path.exists(output_warped_atlas):
            #     reference = f'-ref "{output_warped_structural}"'
            #     # input_atlas = 'infant-2yr-aal-1mm.nii'
            #     input_atlas = 'AAL.nii'  # Assuming this file is in the current directory
            #     interpol_method = '-interp nearestneighbour'
            #     myCommand = f'flirt {reference} -in "{input_atlas}" -out "{output_warped_atlas}" {interpol_method}'
            #     print('Running my Command:', myCommand)
            #     os.system(myCommand)
            #     print('My Command completed:', myCommand)

            # # Make brain tissue masks
            # output_filename_seg = os.path.join(output_folder, output_prefix + '_structSeg_seg.nii.gz')
            # if not os.path.exists(output_filename_seg):
            #     myCommand = f'fast --out="{output_warped_structural_seg}" --type=1 --class=3 "{output_warped_structural}"'
            #     print('Running my Command:', myCommand)
            #     os.system(myCommand)
            #     print('My Command completed:', myCommand)

            # # Copy the input file to the output folder with '_orig.nii' added to the name            
            # output_file_path = os.path.join(output_folder, output_file_name)
            # if not os.path.exists(output_file_path):
            #     shutil.copy(input_path, output_file_path)

            ## Apply warp
            # if not os.path.exists(final_output):
            #     nonlinear_transf = 'cout_fnirt_norm2.nii.gz'
            #     reference = 'MNI152_T1_2mm'
            #     myCommand = f'applywarp --ref="{reference}" --in="{output_norm}" --warp="{nonlinear_transf}" --premat="{affine_output}" --out="{final_output}"'
            #     print('Running my Command:', myCommand)
            #     os.system(myCommand)
            #     print('My Command completed:', myCommand)
            
                           
else:
    print("No folder selected. Exiting...")