import os
import tkinter as tk
from tkinter import filedialog
import shutil
import nibabel as nib
import numpy as np
import math
from nireports.reportlets.nuisance import plot_qi2
from scipy.stats import chi2
from sklearn.neighbors import KernelDensity
import multiprocessing as mp
import subprocess
import sys
from sklearn.neighbors import KernelDensity
from scipy.stats import chi2
import matplotlib.pyplot as plt
from multiprocessing import Pool

import PIL
from PIL import Image
import pandas as pd


# # #### Rigid registration of all low field images to one random one so afterwards mean image can be created

# def process_file(file, input_folder, output_folder):
#     input_path = os.path.join(input_folder, file)
#     output_prefix_tmp = os.path.splitext(file)[0]  
#     output_prefix = os.path.splitext(output_prefix_tmp)[0]  # Extract filename without extension
#     output_reor = os.path.join(output_folder, output_prefix + '_reor.nii.gz')
#     output_no_neck = os.path.join(output_folder, output_prefix + '_noNeck.nii.gz')
#     affine_output = os.path.join(output_folder, output_prefix + '_affineTransf.txt')
#     output_file_name = output_prefix + '_orig.nii.gz'
#     output_affine =  os.path.join(output_folder, output_prefix + '_affine.nii.gz')
#     output_norm =  os.path.join(output_folder, output_prefix + '_affineNorm.nii.gz')

#     # Reorient input image to MNI orientation
#     if not os.path.exists(output_reor):
#         myCommand = f'fslreorient2std "{input_path}" "{output_reor}"'
#         print('Running my Command:', myCommand)
#         os.system(myCommand)
#         print('My Command completed:', myCommand)

#     # Neck removal
#     if not os.path.exists(output_no_neck):
#         myCommand = f'robustfov -i "{output_reor}" -r "{output_no_neck}"'
#         print('Running my Command:', myCommand)
#         os.system(myCommand)
#         print('My Command completed:', myCommand)

#         # Bias correction
#         myCommand = f'fast -B "{output_no_neck}"'
#         print('Running my Command:', myCommand)
#         os.system(myCommand)
#         print('My Command completed:', myCommand)    

#     # Intensity normalization
#     if not os.path.exists(output_norm):
#         myCommand = f'fslstats "{output_no_neck}" -M'
#         print('Running my Command:', myCommand)
#         mean = (subprocess.run(myCommand, shell=True, capture_output=True, text=True)).stdout.strip()
#         # mean = run_command(myCommand)
#         print('My Command completed:', myCommand)  
#         print('Mean:', mean)
        
#         myCommand = f'fslstats "{output_no_neck}" -S'
#         print('Running my Command:', myCommand)
#         std = (subprocess.run(myCommand, shell=True, capture_output=True, text=True)).stdout.strip()
#         # std = run_command(myCommand)
#         print('My Command completed:', myCommand)  
#         print('Standard Deviation:', std)
        
#         myCommand = f'fslmaths "{output_no_neck}" -sub {mean} -div {std} "{output_norm}"'
#         print('Running my Command:', myCommand)
#         os.system(myCommand)
#         print('My Command completed:', myCommand)      

#     # Register brain to MNI template by firstly applying linear registration
#     if not os.path.exists(affine_output):
#         reference = '18448260_ChosenOne'
#         # reference = 'MNI152_T1_2mm_brain'
#         myCommand = f'flirt -ref {reference} -in "{output_norm}" -omat "{affine_output}" -out "{output_affine}"'
#         print('Running my Command:', myCommand)
#         os.system(myCommand)
#         print('My Command completed:', myCommand)

# def main():
#     root = tk.Tk()
#     root.withdraw()

#     input_folder = filedialog.askdirectory(title="Select Input Folder")
#     output_folder = 'lf_12mo_ToRandomOne'

#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
#         print(f"Output folder '{output_folder}' created.")

#     if input_folder:
#         input_files = [file for file in os.listdir(input_folder) if file.endswith('.nii') or file.endswith('.nii.gz')]
        
#         pool = mp.Pool(processes=6)  # Use 6 cores
#         results = [pool.apply_async(process_file, args=(file, input_folder, output_folder)) for file in input_files]

#         pool.close()
#         pool.join()
#         for result in results:
#             result.get()

# if __name__ == "__main__":
#     main()


#### Non-linear registration of all low field images to mean image

# def process_file(file, input_folder, output_folder):
#     input_path = os.path.join(input_folder, file)
#     output_prefix_tmp = os.path.splitext(file)[0]  
#     output_prefix = os.path.splitext(output_prefix_tmp)[0]  # Extract filename without extension
#     affine_output = os.path.join(output_folder, output_prefix + '_affineTransfToAvg.txt')
#     output_affine =  os.path.join(output_folder, output_prefix + '_affineToAvg.nii.gz')
#     output_norm =  os.path.join(output_folder, output_prefix + '_affineNorm.nii.gz')
#     output_warped_structural = os.path.join(output_folder, output_prefix + '_struct.nii.gz')
#     output_warped_structural_seg = os.path.join(output_folder, output_prefix + '_seg.nii.gz')
#     warpfield_output = os.path.join(output_folder, output_prefix + '_warpfield.nii.gz')   

#     # Register brain to MNI template by firstly applying linear registration
#     # if not os.path.exists(affine_output):
#     reference = 'average_T1_12MO.nii'
#     # reference = 'MNI152_T1_2mm_brain'
#     myCommand = f'flirt -ref {reference} -in "{output_norm}" -omat "{affine_output}" -out "{output_affine}"'
#     print('Running my Command:', myCommand)
#     os.system(myCommand)
#     print('My Command completed:', myCommand)

#     # Register brain to MNI template by secondly applying non-linear b-spline registration
#     # if not os.path.exists(warpfield_output):
#     reference2 = 'average_T1_12MO.nii'
#     configuration = '--config=parameters_test'
#     myCommand = f'fnirt --in="{output_norm}" --ref="{reference2}" --aff="{affine_output}" --fout="{warpfield_output}" --iout="{output_warped_structural}" --subsamp=8,4,2,2'
#     print('Running my Command:', myCommand)
#     os.system(myCommand)
#     print('My Command completed:', myCommand)

#     output_filename_seg = os.path.join(output_folder, output_prefix + '_seg_seg.nii.gz')
#     if not os.path.exists(output_filename_seg):
#         myCommand = f'fast --out="{output_warped_structural_seg}" --type=1 --class=3 "{output_warped_structural}"'
#         print('Running my Command:', myCommand)
#         os.system(myCommand)
#         print('My Command completed:', myCommand)

# def main():
#     root = tk.Tk()
#     root.withdraw()

#     input_folder = filedialog.askdirectory(title="Select Input Folder")
#     output_folder = 'lf_12mo_ToRandomOne'

#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
#         print(f"Output folder '{output_folder}' created.")

#     if input_folder:
#         input_files = [file for file in os.listdir(input_folder) if file.endswith('.nii') or file.endswith('.nii.gz')]
        
#         pool = mp.Pool(processes=6)  # Use 6 cores
#         results = [pool.apply_async(process_file, args=(file, input_folder, output_folder)) for file in input_files]

#         pool.close()
#         pool.join()
#         for result in results:
#             result.get()

# if __name__ == "__main__":
#     main()


############

# #### Rigid registration of all low field images to one random one so afterwards mean image can be created

def apply_mask(nifti_file, mask_file, output_file):
    # Load the NIfTI file
    nii = nib.load(nifti_file)
    nii_data = nii.get_fdata()

    # Load the mask file
    mask = nib.load(mask_file)
    mask_data = mask.get_fdata()

    # Ensure the mask has the same shape as the NIfTI data
    if nii_data.shape != mask_data.shape:
        raise ValueError("The mask must have the same dimensions as the NIfTI file.")

    # Apply the mask
    masked_data = nii_data * mask_data

    # Create a new NIfTI image
    masked_nii = nib.Nifti1Image(masked_data, nii.affine, nii.header)

    # Save the masked NIfTI file
    nib.save(masked_nii, output_file)
    print(f"Masked NIfTI file saved to {output_file}")

def process_file(file, input_folder, output_folder):
    input_path = os.path.join(input_folder, file)
    output_prefix_tmp = os.path.splitext(file)[0]  
    output_prefix = os.path.splitext(output_prefix_tmp)[0]  # Extract filename without extension
    output_reor = os.path.join(output_folder, output_prefix + '_reor.nii.gz')
    output_no_neck = os.path.join(output_folder, output_prefix + '_noNeck.nii.gz')
    affine_output = os.path.join(output_folder, output_prefix + '_affineTransf.txt')
    output_file_name = output_prefix + '_orig.nii.gz'
    output_affine =  os.path.join(output_folder, output_prefix + '_affine.nii.gz')
    output_norm =  os.path.join(output_folder, output_prefix + '_Norm.nii.gz')
    output_warped_structural = os.path.join(output_folder, output_prefix + '_struct.nii.gz')
    output_warped_structural_brain = os.path.join(output_folder, output_prefix + '_brain.nii.gz')
    output_warped_structural_seg = os.path.join(output_folder, output_prefix + '_seg.nii.gz')
    warpfield_output = os.path.join(output_folder, output_prefix + '_warpfield.nii.gz')

    # Reorient input image to MNI orientation
    if not os.path.exists(output_reor):
        myCommand = f'fslreorient2std "{input_path}" "{output_reor}"'
        print('Running my Command:', myCommand)
        os.system(myCommand)
        print('My Command completed:', myCommand)

    # Neck removal
    if not os.path.exists(output_no_neck):
        myCommand = f'robustfov -i "{output_reor}" -r "{output_no_neck}"'
        print('Running my Command:', myCommand)
        os.system(myCommand)
        print('My Command completed:', myCommand)

        # Bias correction
        myCommand = f'fast -B "{output_no_neck}"'
        print('Running my Command:', myCommand)
        os.system(myCommand)
        print('My Command completed:', myCommand)    

    # Intensity normalization
    if not os.path.exists(output_norm):
        # Calculate minimum intensity
        myCommand = f'fslstats "{output_no_neck}" -R'
        print('Running my Command:', myCommand)
        min_max = (subprocess.run(myCommand, shell=True, capture_output=True, text=True)).stdout.strip()
        min_val, max_val = map(float, min_max.split())
        print('My Command completed:', myCommand)  
        print('Min:', min_val)
        print('Max:', max_val)
        
        # Apply Min-Max normalization
        myCommand = f'fslmaths "{output_no_neck}" -sub {min_val} -div {max_val - min_val} "{output_norm}"'
        print('Running my Command:', myCommand)
        os.system(myCommand)
        print('My Command completed:', myCommand) 

    # Register brain to MNI template by firstly applying linear registration
    if not os.path.exists(affine_output):
        reference = 'average_T1_12MO.nii'
        # reference = 'MNI152_T1_2mm_brain'
        myCommand = f'flirt -ref {reference} -in "{output_norm}" -omat "{affine_output}" -out "{output_affine}"'
        print('Running my Command:', myCommand)
        os.system(myCommand)
        print('My Command completed:', myCommand)

    # Register brain to MNI template by secondly applying non-linear b-spline registration
    if not os.path.exists(warpfield_output):
        reference2 = 'average_T1_12MO.nii'
        configuration = '--config=parameters_test'
        myCommand = f'fnirt --in="{output_norm}" --ref="{reference2}" --aff="{affine_output}" --fout="{warpfield_output}" --iout="{output_warped_structural}" --subsamp=8,4,2,2'
        print('Running my Command:', myCommand)
        os.system(myCommand)
        print('My Command completed:', myCommand)

    output_filename_seg = os.path.join(output_folder, output_prefix + '_seg_seg.nii.gz')
    # if not os.path.exists(output_filename_seg):
    mask_file = 'average_T1_12MO_brain_mask.nii.gz'
    nifti_file = output_warped_structural
    output_file = output_warped_structural_brain
    apply_mask(nifti_file, mask_file, output_file)

    myCommand = f'fast --out="{output_warped_structural_seg}" --type=1 --class=3 "{output_warped_structural_brain}"'
    print('Running my Command:', myCommand)
    os.system(myCommand)
    print('My Command completed:', myCommand)

def main():
    root = tk.Tk()
    root.withdraw()

    input_folder = filedialog.askdirectory(title="Select Input Folder")
    output_folder = 'lf_12mo_ToRAverage'

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Output folder '{output_folder}' created.")

    if input_folder:
        input_files = [file for file in os.listdir(input_folder) if file.endswith('.nii') or file.endswith('.nii.gz')]
        
        pool = mp.Pool(processes=3)  # Use 6 cores
        results = [pool.apply_async(process_file, args=(file, input_folder, output_folder)) for file in input_files]

        pool.close()
        pool.join()
        for result in results:
            result.get()

if __name__ == "__main__":
    main()
