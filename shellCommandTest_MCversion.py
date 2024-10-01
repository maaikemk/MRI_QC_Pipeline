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

import sys
from sklearn.neighbors import KernelDensity
from scipy.stats import chi2
import matplotlib.pyplot as plt
from multiprocessing import Pool

# def process_file(file, input_folder, output_folder):
#     input_path = os.path.join(input_folder, file)
#     output_prefix_tmp = os.path.splitext(file)[0]
#     output_prefix = os.path.splitext(output_prefix_tmp)[0]  # Extract filename without extension
#     output_reor = os.path.join(output_folder, output_prefix + '_reor.nii.gz')
#     output_no_neck = os.path.join(output_folder, output_prefix + '_noNeck.nii.gz')
#     output_extracted = os.path.join(output_folder, output_prefix + '_extracted.nii.gz')
#     output_norm = os.path.join(output_folder, output_prefix + '_norm.nii.gz')
#     output_warped_structural = os.path.join(output_folder, output_prefix + '_struct.nii.gz')
#     output_warped_atlas = os.path.join(output_folder, output_prefix + '_warpedAtlas.nii.gz')
#     output_warped_structural_seg = os.path.join(output_folder, output_prefix + '_structSeg.nii.gz')
#     warp_output = os.path.join(output_folder, output_prefix + '_nonlineartransf.nii.gz')
#     warpfield_output = os.path.join(output_folder, output_prefix + '_warpfield.nii.gz')
#     affine_output = os.path.join(output_folder, output_prefix + '_affineTransf.txt')
#     output_file_name = output_prefix + '_orig.nii.gz'

#     if not os.path.exists(output_reor):
#         myCommand = f'fslreorient2std "{input_path}" "{output_reor}"'
#         print('Running my Command:', myCommand)
#         os.system(myCommand)
#         print('My Command completed:', myCommand)

#     if not os.path.exists(output_no_neck):
#         myCommand = f'robustfov -i "{output_reor}" -r "{output_no_neck}"'
#         print('Running my Command:', myCommand)
#         os.system(myCommand)
#         print('My Command completed:', myCommand)

#     if not os.path.exists(output_extracted):
#         myCommand = f'bet "{output_no_neck}" "{output_extracted}" -A'
#         print('Running my Command:', myCommand)
#         os.system(myCommand)
#         print('My Command completed:', myCommand)

#     if not os.path.exists(affine_output):
#         reference = 'infant_1yr_intensity_1mm'
#         myCommand = f'flirt -ref {reference} -in "{output_extracted}" -omat "{affine_output}"'
#         print('Running my Command:', myCommand)
#         os.system(myCommand)
#         print('My Command completed:', myCommand)

#     if not os.path.exists(output_warped_structural):
#         reference2 = 'infant_1yr_intensity_1mm'
#         myCommand = f'fnirt --in="{output_extracted}" --ref="{reference2}" --aff="{affine_output}" --fout="{warpfield_output}" --iout="{output_warped_structural}" --subsamp=8,4,2,2'
#         print('Running my Command:', myCommand)
#         os.system(myCommand)
#         print('My Command completed:', myCommand)

#     if not os.path.exists(output_warped_atlas):
#         reference = f'-ref "{output_warped_structural}"'
#         input_atlas = 'infant-1yr-aal-1mm.nii'
#         interpol_method = '-interp nearestneighbour'
#         myCommand = f'flirt {reference} -in "{input_atlas}" -out "{output_warped_atlas}" {interpol_method}'
#         print('Running my Command:', myCommand)
#         os.system(myCommand)
#         print('My Command completed:', myCommand)

#     output_filename_seg = os.path.join(output_folder, output_prefix + '_structSeg_seg.nii.gz')
#     if not os.path.exists(output_filename_seg):
#         myCommand = f'fast --out="{output_warped_structural_seg}" --type=1 --class=3 "{output_warped_structural}"'
#         print('Running my Command:', myCommand)
#         os.system(myCommand)
#         print('My Command completed:', myCommand)

#     output_file_path = os.path.join(output_folder, output_file_name)
#     if not os.path.exists(output_file_path):
#         shutil.copy(input_path, output_file_path)

# def main():
#     root = tk.Tk()
#     root.withdraw()

#     input_folder = filedialog.askdirectory(title="Select Input Folder")
#     output_folder = 'MetaOutputs'

#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
#         print(f"Output folder '{output_folder}' created.")

#     if input_folder:
#         input_files = [file for file in os.listdir(input_folder) if file.endswith('.nii') or file.endswith('.nii.gz')]
        
#         pool = mp.Pool(processes=2)  # Use 6 cores
#         results = [pool.apply_async(process_file, args=(file, input_folder, output_folder)) for file in input_files]

#         pool.close()
#         pool.join()
#         for result in results:
#             result.get()

# if __name__ == "__main__":
#     main()


########### Image analysis

# def save_nifti(img, filename, postfix, output_folder, affine_matr):
#     """
#     Saves 3D arrays as NIfTI images with a certain file name format.
#     """
#     nifti_array = np.array(img, dtype=np.float32)
#     nifti_array = nib.Nifti1Image(nifti_array, affine=affine_matr)
#     nib.save(nifti_array, output_folder + '/' + filename + postfix) 

# def make_snrmap(img, filename, output_folder, affine_matr, region_values, atlas_data, std_air):
#     """
#     Calculates snr, snrD, signal, and std per atlas region of the brain and saves these 
#     in NIfTI files and in txt files.
#     """
#     X, Y, Z = img.shape
#     mean_signal = []
#     std_signal = []
#     snrlist = []
#     snrDlist = []
#     snr_map = np.zeros((X, Y, Z))
#     signal_map = np.zeros((X, Y, Z))
#     std_map = np.zeros((X, Y, Z))
#     snr_d_map = np.zeros((X, Y, Z))
#     postfixes = ['_snr', '_signal', '_std', '_snrD']

#     # Check if the files with postfixes exist in the output folder
#     files_exist = [os.path.exists(os.path.join(output_folder, f"{filename}{postfix}.nii")) for postfix in postfixes]
#     if all(files_exist):
#         return

#     print(f"Generating snr(D), signal, and std maps of {filename}")
#     for value in range(1, len(region_values)):
#         gray_value = int(region_values[value])
#         total_signal = []
#         voxelcoordx = []
#         voxelcoordy = []
#         voxelcoordz = []
#         for x in range(X):
#             for y in range(Y):
#                 for z in range(Z):
#                     voxelvalue = atlas_data[x, y, z]
#                     if voxelvalue == gray_value:
#                         total_signal.append(img[x, y, z])
#                         voxelcoordx.append(x)
#                         voxelcoordy.append(y)
#                         voxelcoordz.append(z)
        
#         mean_signal_reg = round(np.mean(total_signal), 2)
#         mean_signal.append(mean_signal_reg)
#         std_signal_reg = round(np.std(total_signal), 2)
#         std_signal.append(std_signal_reg)
#         snr_reg = round(mean_signal_reg / (std_signal_reg * (len(voxelcoordx) / (len(voxelcoordx) - 1)) ** 0.5), 2)
#         snrD_reg = round(mean_signal_reg / ((2 / (4 - math.pi)) ** 0.5 * std_air), 2)
#         snrlist.append(snr_reg)
#         snrDlist.append(snrD_reg)
#         print('tic')
#         for coord in range(len(voxelcoordx)):
#             snr_map[voxelcoordx[coord], voxelcoordy[coord], voxelcoordz[coord]] = snr_reg
#             signal_map[voxelcoordx[coord], voxelcoordy[coord], voxelcoordz[coord]] = round(mean_signal_reg, 2)
#             std_map[voxelcoordx[coord], voxelcoordy[coord], voxelcoordz[coord]] = round(std_signal_reg, 2) 
#             snr_d_map[voxelcoordx[coord], voxelcoordy[coord], voxelcoordz[coord]] = snrD_reg
        
#     print('toc')
#     save_nifti(snr_map, filename, '_snr', output_folder, affine_matr)
#     save_nifti(signal_map, filename, '_signal', output_folder, affine_matr)
#     save_nifti(std_map, filename, '_std', output_folder, affine_matr)
#     save_nifti(snr_d_map, filename, '_snrD', output_folder, affine_matr)
#     write_zmapfile(filename, output_folder, region_values, snrlist, '_snrZmap.txt')
#     write_zmapfile(filename, output_folder, region_values, snrDlist, '_snrDZmap.txt')

# def calculate_cjv(img,wm_mask,gm_mask):
#     """
#     Calculates the Coefficient of Joint Variation (CJV) for an image, using white matter (WM)
#     and gray matter (GM) masks.
    
#     img: 3D numpy array, input image data
#     wm_mask: 3D numpy array, binary mask for white matter
#     gm_mask: 3D numpy array, binary mask for gray matter
    
#     Return: 
#     cjv: float
#     """
#     img_wm = img*wm_mask
#     img_gm = img*gm_mask
#     std_wm = img_wm[img_wm!=0].std()
#     mean_wm = img_wm[img_wm!=0].mean()
#     std_gm = img_gm[img_gm!=0].std()
#     mean_gm = img_gm[img_gm!=0].mean()
#     cjv = (std_wm + std_gm) / abs(mean_wm - mean_gm)
#     return cjv

# def calculate_cnr(img,wm_mask,gm_mask,air_std):
#     """
#     Calculates the Contrast-to-Noise Ratio (CNR) for an image, using white matter (WM) and
#     gray matter (GM) masks, and the standard deviation of background air noise.
    
#     img: 3D numpy array, input image data
#     wm_mask: 3D numpy array, binary mask for white matter
#     gm_mask: 3D numpy array, binary mask for gray matter
#     air_std: float, standard deviation of background air noise
    
#     Returns: 
#     cnr: float
#     """
#     img_wm = img*wm_mask
#     img_gm = img*gm_mask
#     std_wm = img_wm[img_wm!=0].std()
#     mean_wm = img_wm[img_wm!=0].mean()
#     std_gm = img_gm[img_gm!=0].std()
#     mean_gm = img_gm[img_gm!=0].mean()
#     cnr = abs(mean_wm - mean_gm) / (air_std**2 + std_wm**2 + std_gm**2)**0.5
#     return cnr

# def art_qi1(airmask, artmask):
#     """
#     Detects artifacts in an image using the method described in [Mortamet2009].
#     Calculates QI1 as the proportion of voxels with intensity corrupted by 
#     artifacts normalized by the number of voxels in the background mask. 
#     Near-zero values are better.
    
#     airmask: 3D numpy array, input air mask without artifacts
#     artmask: 3D numpy array, input artifacts mask
    
#     Returns: the quality index QI1 as a float.
#     """
#     if airmask.sum() < 1:
#         return -1.0

#     # Count the ratio between artifacts and the total voxels in the background mask
#     return float(artmask.sum() / (airmask.sum() + artmask.sum()))

# def art_qi2(img,airmask,min_voxels=int(1e3),max_voxels=int(3e5),save_plot=False,coil_elements=32):
#     """
#     Calculates the quality index QI2 based on the goodness-of-fit of a centered chi-square distribution
#     onto the intensity distribution of non-artifactual background within the background mask using the 
#     method described in [Mortamet2009]. Near-zero values are better.

#     img: 3D numpy array, input image data
#     airmask: 3D numpy array, input air mask without artifacts
#     min_voxels: int, minimum number of voxels for analysis (default 1000)
#     max_voxels: int, maximum number of voxels for analysis (default 300000)
#     save_plot: bool, whether to save a plot of the fitting (default False)
#     coil_elements: int, number of coil elements (default 32)
    
#     Returns: 
#     gof: float, goodness-of-fit
#     """

#     # S. Ogawa was born
#     np.random.seed(1191935)

#     data = np.nan_to_num(img[airmask > 0], posinf=0.0)
#     data[data < 0] = 0

#     # Write out figure of the fitting
#     out_file = op.abspath("error.svg")
#     with open(out_file, "w") as ofh:
#         ofh.write("<p>Background noise fitting could not be plotted.</p>")

#     if (data > 0).sum() < min_voxels:
#         return 0.0, out_file

#     data *= 100 / np.percentile(data, 99)
#     modelx = data if len(data) < max_voxels else np.random.choice(data, size=max_voxels)

#     x_grid = np.linspace(0.0, 110, 1000)

#     # Estimate data pdf with KDE on a random subsample
#     kde_skl = KernelDensity(kernel="gaussian", bandwidth=4.0).fit(modelx[:, np.newaxis])
#     kde = np.exp(kde_skl.score_samples(x_grid[:, np.newaxis]))

#     # Find cutoff
#     kdethi = np.argmax(kde[::-1] > kde.max() * 0.5)

#     # Fit X^2
#     param = chi2.fit(modelx, coil_elements)
#     chi_pdf = chi2.pdf(x_grid, *param[:-2], loc=param[-2], scale=param[-1])

#     # Compute goodness-of-fit (gof)
#     gof = float(np.abs(kde[-kdethi:] - chi_pdf[-kdethi:]).mean())
#     if save_plot:
#         out_file = plot_qi2(x_grid, kde, chi_pdf, modelx, kdethi)

#     return gof

# def make_airmask(bg_data, noNeck_data):
#     """
#     Creates an image containing only the background noise from the background mask and 
#     image data, and calculates the standard deviation of the background noise.

#     bg_data: 3D numpy array, mask containing the whole head
#     noNeck_data: 3D numpy array, MRI image (reoriented and with the neck cut off)
    
#     Returns: 
#     air_data: 3D numpy array, the background noise 
#     std_signal_air: float, the standard deviation of the background noise.
#     """
#     air_data = (1 - bg_data) * noNeck_data
#     std_signal_air = round(air_data[air_data!=0].std(),2)
#     return air_data, std_signal_air

# def make_artimg(air_data):
#     """
#     Generates an artifacts image from air data using the 
#     method described in [Mortamet2009].

#     air_data: 3D numpy array, containg background noise
    
#     Returns:
#     art_img: 3D numpy array, the artifacts image.
#     """
#     X,Y,Z = air_data.shape
#     art_img = np.zeros((X,Y,Z))
#     air_data_tmp = air_data[air_data!=0].flatten()
#     hist_amp, hist_value = np.histogram(air_data_tmp,335)
#     noise_trh = hist_value[np.where(hist_amp == hist_amp.max())]

#     for x in range(X):
#         for y in range(Y):
#             for z in range(Z):
#                 voxelvalue = air_data[x,y,z]
#                 if air_data[x,y,z] <= noise_trh:
#                     art_img[x,y,z] = 0
#                 else:
#                     art_img[x,y,z] = 1
#     return art_img

# def make_wm_gm_masks(seg_data):
#     """
#     Creates white matter (WM) and gray matter (GM) masks from segmented data where 
#     white matter has the label '3' and grey matter has the label '2'.

#     seg_data: 3D numpy array, segmented data 
    
#     Returns:
#     wm_mask: 3D numpy array, the binary WM mask.
#     gm_mask: 3D numpy array, the binary GM mask. 
#     """
#     X, Y, Z = seg_data.shape
#     wm_mask = np.zeros((X,Y,Z))
#     gm_mask = np.zeros((X,Y,Z))
#     brainmask = np.zeros((X,Y,Z))
#     for x in range(X):
#         for y in range(Y):
#             for z in range(Z):
#                 if seg_data[x,y,z] == 3:
#                     wm_mask[x,y,z] = 1
#                 elif seg_data[x,y,z] == 2:
#                     gm_mask[x,y,z] = 1
#                 if seg_data[x,y,z] > 0:
#                     brainmask[x,y,z] = 1
#     return wm_mask, gm_mask, brainmask

# def read_brainregionsfile(brainregionsfile):
#     """
#     Reads brain regions data from a txt file with a certain format.

#     brainregionsfile: str, path to the brain regions txt file
    
#     Returns:
#     region_values: list of the brain region numeral labels
#     region_names: list of brain region names
#     """
#     with open(brainregionsfile) as brainregions_file:
#         brainregions = brainregions_file.read()

#     brainregions_col = brainregions.split('\n')
#     columns = [c.split()  for c in brainregions_col]
#     region_values = [c[1] for c in columns]
#     region_names = [c[0] for c in columns]
#     return region_values,region_names

# def write_qmfile(filename,output_folder,cnr,cjv,qi1,qi2,sum_def):
#     """
#     Writes quality metrics to a txt file.

#     filename: str, identifier of the input file name
#     output_folder: str, path to the output folder 
#     cnr: float, contrast-to-noise ratio
#     cjv: float, coefficient of joint variation
#     qi1: float, quality index 1
#     qi2: float, quality index 2
#     sum_def: float, sum of deformation field
    
#     Returns None
#     """
#     with open(output_folder + '/' + filename+'_qm.txt','w') as quality_metrics:
#         quality_metrics.write('CNR: ' + str(cnr) + '\n' + 'CJV: ' + str(cjv) + '\n' + 'QI1: ' + str(qi1) + '\n' + 'QI2: ' + str(qi2) + '\n' + 'SumDefField: ' + str(sum_def))

# def write_zmapfile(filename,output_folder,region_names,list,postfix):
#     """
#     Writes a txt file containing brain region names and corresponding values

#     filename: str, identifier of the input file name
#     output_folder: str, path to the output folder
#     region_names: list of str, names of brain regions
#     list: list of float, corresponding values for each brain region
#     postfix: str, postfix for the output file

#     Returns None
#     """
#     with open(output_folder + '/' + filename+postfix,'a') as Zmap:
#         for idx in range(len(region_names)-1):
#             Zmap.write(str(region_names[idx+1]) + ' ' + str(list[idx]) + '\n')

# def compare_list_lengths(lists):
#     """
#     Compares the lengths of multiple lists and checks if they are all equal.

#     lists: list of lists, the lists to compare

#     Raises:
#     SystemExit: if the lengths of the lists are not equal
#     """
#     lengths = [len(lst) for lst in lists]
#     print(lengths)
#     if len(set(lengths)) != 1:
#         print("Stopping: Image data incomplete!")
#         sys.exit(1)
#     else:
#         print("Analysing image data")

# def def_field_visual(warpfield,img_data,brainmask,alpha=0.6):
#     """
#     warpfield: 4D numpy array, the deformation field data
#     img_data: 3D numpy array, input image data for interpolation
#     ###brainmask_data:
#     alpha: float, transparency of the visualization (default: 0.6)

#     Returns:
#     sum_def: float, the sum of the deformation field
#     out_img: 4D numpy array (x,y,z,channels), the deformation field image including the brain defined in opacity
#     newwarpfield: 3D numpy array, the absolute value of the deformation field
#     """
#     X, Y, Z, A = warpfield.shape
#     sum_def = 0

#     newwarpfield = np.zeros((X,Y,Z))
#     for x in range(X):
#         for y in range(Y):
#             for z in range(Z):
#                 for a in range(A):
#                     sum_def += abs(warpfield[x,y,z,a])
#                     newwarpfield[x,y,z] += abs(warpfield[x,y,z,a])

#     # abs_warpfield = np.interp(newwarpfield, (newwarpfield.min(), newwarpfield.max()), (0, +1))
#     # target_img = np.interp(img_data, (img_data.min(), img_data.max()), (0, +1))

#     # # Apply the Inferno colormap to the grayscale warpfield
#     # cm = plt.get_cmap('inferno')
#     # warpfield_image = cm(abs_warpfield)
#     # warpfield_image[:,:,:,3] = alpha

#     # rgba_image = np.zeros((*target_img.shape, 4))
#     # rgba_image[..., 3] = 1 - target_img
#     # out_img = warpfield_image + ((1 - alpha) * rgba_image[:, :, :, :])

#     out_img = newwarpfield * brainmask
#     outZ_img = newwarpfield
    
#     return sum_def, out_img, outZ_img


# def process_file(file_data):
#     """
#     Process a single file and its associated data.
#     """
#     file, file_outskin, file_noNeck, file_seg, file_name, file_atlas, file_field, file_brainmask, output_folder2, region_values, affine_matr = file_data
#     img_data = nib.load(file).get_fdata()
#     bg_data = nib.load(file_outskin).get_fdata()
#     noNeck_data = nib.load(file_noNeck).get_fdata()
#     seg_data = nib.load(file_seg).get_fdata()
#     atlas_data = nib.load(file_atlas).get_fdata()
#     field_data = nib.load(file_field).get_fdata()

#     air_img, std_air = make_airmask(bg_data, noNeck_data)
#     wm_mask, gm_mask, brainmask = make_wm_gm_masks(seg_data)
#     art_img = make_artimg(air_img)
#     make_snrmap(img_data, file_name, output_folder2, affine_matr, region_values, atlas_data, std_air)
#     sum_def, out_img, newwarpfield = def_field_visual(field_data, img_data, brainmask)

#     if not os.path.exists(os.path.join(output_folder2, f"{file_name}_field.nii")):
#         save_nifti(out_img, file_name, '_field', output_folder2, affine_matr)
#         save_nifti(newwarpfield, file_name, '_fieldZmap', output_folder2, affine_matr)

#     if not os.path.exists(os.path.join(output_folder2, f"{file_name}_brainmask.nii")):
#         save_nifti(brainmask, file_name, '_brainmask', output_folder2, affine_matr)

#     if not os.path.exists(os.path.join(output_folder2, f"{file_name}_qm.txt")):
#         write_qmfile(file_name, output_folder2, calculate_cjv(img_data, wm_mask, gm_mask), calculate_cnr(img_data, wm_mask, gm_mask, std_air), art_qi1(air_img, art_img), art_qi1(air_img, art_img) + art_qi2(noNeck_data, air_img), sum_def)

# def main():
#     folder_path = 'MetaOutputs'
#     output_folder2 = 'MetaOutputsGUI'

#     if not os.path.exists(output_folder2):
#         os.makedirs(output_folder2)
#         print(f"Output folder '{output_folder2}' created.")

#     region_values, region_names = read_brainregionsfile('brainregions_infant.txt')

#     files_list = []
#     files_outskin = []
#     files_noNeck = []
#     files_seg = []
#     file_names = []
#     files_atlas = []
#     files_field = []
#     files_brainmask = []

#     if os.path.exists(folder_path):
#         for file_name in os.listdir(folder_path):
#             if file_name.endswith('_struct.nii.gz'):
#                 files_list.append(os.path.join(folder_path, file_name))
#                 namelist = file_name.split("_")
#                 identifier = namelist[2]
#                 file_names.append(identifier)
#                 new_path = os.path.join(output_folder2, identifier + '_struct.nii.gz')
#                 if not os.path.exists(new_path):
#                     shutil.copy(os.path.join(folder_path, file_name), os.path.join(output_folder2, file_name))
#                     shutil.move(os.path.join(output_folder2, file_name), new_path)

#             elif file_name.endswith('_outskin_mask.nii.gz'):
#                 files_outskin.append(os.path.join(folder_path, file_name))

#             elif file_name.endswith('_noNeck.nii.gz'):
#                 files_noNeck.append(os.path.join(folder_path, file_name))

#             elif file_name.endswith('_structSeg_seg.nii.gz'):
#                 files_seg.append(os.path.join(folder_path, file_name))

#             elif file_name.endswith('_warpedAtlas.nii.gz'):
#                 files_atlas.append('AAL.nii')

#             elif file_name.endswith('_warpfield.nii.gz'):
#                 files_field.append(os.path.join(folder_path, file_name))

#             elif file_name.endswith('_inskull_mask.nii.gz'):
#                 files_brainmask.append(os.path.join(folder_path, file_name))

#     lists = [files_list, files_outskin, files_noNeck, files_seg, file_names, files_atlas, files_field, files_brainmask]
#     compare_list_lengths(lists)

#     file_data_list = [
#         (files_list[i], files_outskin[i], files_noNeck[i], files_seg[i], file_names[i], files_atlas[i], files_field[i], files_brainmask[i], output_folder2, region_values, nib.load(files_list[i]).affine)
#         for i in range(len(files_list))
#     ]

#     # with Pool(6) as pool:
#     #     pool.map(process_file, file_data_list)

#     pool = mp.Pool(processes=4)  # Use 6 cores
#     results = [pool.apply_async(process_file, args=(file_data_list[i])) for i in range(len(files_list))]

#     pool.close()
#     pool.join()
#     for result in results:
#         result.get()

# if __name__ == '__main__':
#     main()

