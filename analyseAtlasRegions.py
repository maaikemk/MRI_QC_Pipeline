import nilearn.datasets 
import nibabel
import numpy as np
import statistics
from sklearn.preprocessing import binarize
from sklearn import preprocessing
import sklearn
import matplotlib.pyplot as plt
import math
from nireports.reportlets.nuisance import plot_qi2
from scipy.stats import chi2
from sklearn.neighbors import KernelDensity
import os.path as op
import PIL
from PIL import Image
import scipy
import skimage
import pandas as pd
import os  
import sys
import shutil
  
folder_path = r'C:\Users\maaik\OneDrive\Documents\UU\MinorResearchProject\Stefan\shellCommandTest\RealWrapperData\MetaOutputs'  # Replace this with the actual path to your folder

def save_nifti(img,filename,postfix,affine_matr,output_folder):
    nifti_array = np.array(img, dtype=np.float32)
    nifti_array = nibabel.Nifti1Image(nifti_array, affine=affine_matr)
    nibabel.save(nifti_array, output_folder + '/' + filename + postfix) 

def make_snrmap(img,region_values,atlas_data,filename,affine_matr,std_air, output_folder):
    X, Y, Z = img.shape
    mean_signal = []
    std_signal = []
    snr_map = np.zeros((X,Y,Z))
    signal_map = np.zeros((X,Y,Z))
    std_map = np.zeros((X,Y,Z))
    snr_d_map = np.zeros((X,Y,Z))
    postfixes = ['_snr', '_signal', '_std', '_snrD']

    # Check if the files with postfixes exist in the output folder
    files_exist = [os.path.exists(os.path.join(output_folder, f"{filename}{postfix}.nii.gz")) for postfix in postfixes]
    if all(files_exist):
        print("snr(D), signal, and std maps already exist. Skipping generation.")
        return

    for value in range(1,len(region_values)):
        gray_value = int(region_values[value])
        total_signal = []
        voxelcoordx = []
        voxelcoordy = []
        voxelcoordz = []
        for x in range(X):
            for y in range(Y):
                for z in range(Z):
                    voxelvalue = atlas_data[x,y,z]
                    if voxelvalue == gray_value:
                        total_signal.append(img[x,y,z])
                        voxelcoordx.append(x)
                        voxelcoordy.append(y)
                        voxelcoordz.append(z)
    
        mean_signal_reg = round(statistics.mean(total_signal),2)
        mean_signal.append(mean_signal_reg)
        std_signal_reg = round(statistics.stdev(total_signal),2)
        std_signal.append(std_signal_reg)
        # print("The SNR in region",region_names[value],"is",round(mean_signal_reg/(std_signal_reg * (len(voxelcoordx)/(len(voxelcoordx)-1))**0.5),2))
        for coord in range(len(voxelcoordx)):
            snr_map[voxelcoordx[coord],voxelcoordy[coord],voxelcoordz[coord]] = round(mean_signal_reg/(std_signal_reg * (len(voxelcoordx)/(len(voxelcoordx)-1))**0.5),2)
            signal_map[voxelcoordx[coord],voxelcoordy[coord],voxelcoordz[coord]] = round(mean_signal_reg,2)
            std_map[voxelcoordx[coord],voxelcoordy[coord],voxelcoordz[coord]] = round(std_signal_reg,2) 
            snr_d_map[voxelcoordx[coord],voxelcoordy[coord],voxelcoordz[coord]] = round(mean_signal_reg/((2/(4-math.pi))**0.5*std_air),2)
    
    save_nifti(snr_map,filename,'_snr',affine_matr,output_folder)
    save_nifti(signal_map,filename,'_signal',affine_matr,output_folder)
    save_nifti(std_map,filename,'_std',affine_matr,output_folder)
    save_nifti(snr_d_map,filename,'_snrD',affine_matr,output_folder)

def calculate_cjv(img,wm_mask,gm_mask):
    img_wm = img*wm_mask
    img_gm = img*gm_mask
    std_wm = img_wm[img_wm!=0].std()
    mean_wm = img_wm[img_wm!=0].mean()
    std_gm = img_gm[img_gm!=0].std()
    mean_gm = img_gm[img_gm!=0].mean()
    return (std_wm + std_gm) / abs(mean_wm - mean_gm)

def calculate_cnr(img,wm_mask,gm_mask,air_std):
    img_wm = img*wm_mask
    img_gm = img*gm_mask
    std_wm = img_wm[img_wm!=0].std()
    mean_wm = img_wm[img_wm!=0].mean()
    std_gm = img_gm[img_gm!=0].std()
    mean_gm = img_gm[img_gm!=0].mean()
    return abs(mean_wm - mean_gm) / (air_std**2 + std_wm**2 + std_gm**2)**0.5

def art_qi1(airmask, artmask):
    r"""
    Detect artifacts in the image using the method described in [Mortamet2009]_.
    Calculates :math:`\text{QI}_1`, as the proportion of voxels with intensity
    corrupted by artifacts normalized by the number of voxels in the "*hat*"
    mask (i.e., the background region above the nasio-occipital plane):

    .. math ::

        \text{QI}_1 = \frac{1}{N} \sum\limits_{x\in X_\text{art}} 1

    Near-zero values are better.
    If :math:`\text{QI}_1 = -1`, then the "*hat*" mask (background) was empty
    and the dataset is likely a skull-stripped image or has been heavily
    post-processed.

    :param numpy.ndarray airmask: input air mask, without artifacts
    :param numpy.ndarray artmask: input artifacts mask

    """
    if airmask.sum() < 1:
        return -1.0

    # Count the ratio between artifacts and the total voxels in "hat" mask
    return float(artmask.sum() / (airmask.sum() + artmask.sum()))

def art_qi2(
    img,
    airmask,
    min_voxels=int(1e3),
    max_voxels=int(3e5),
    save_plot=False,
    coil_elements=32,
):
    r"""
    Calculates :math:`\text{QI}_2`, based on the goodness-of-fit of a centered
    :math:`\chi^2` distribution onto the intensity distribution of
    non-artifactual background (within the "hat" mask):


    .. math ::

        \chi^2_n = \frac{2}{(\sigma \sqrt{2})^{2n} \, (n - 1)!}x^{2n - 1}\, e^{-\frac{x}{2}}

    where :math:`n` is the number of coil elements.

    :param numpy.ndarray img: input data
    :param numpy.ndarray airmask: input air mask without artifacts

    """

    # S. Ogawa was born
    np.random.seed(1191935)

    data = np.nan_to_num(img[airmask > 0], posinf=0.0)
    data[data < 0] = 0

    # Write out figure of the fitting
    out_file = op.abspath("error.svg")
    with open(out_file, "w") as ofh:
        ofh.write("<p>Background noise fitting could not be plotted.</p>")

    if (data > 0).sum() < min_voxels:
        return 0.0, out_file

    data *= 100 / np.percentile(data, 99)
    modelx = data if len(data) < max_voxels else np.random.choice(data, size=max_voxels)

    x_grid = np.linspace(0.0, 110, 1000)

    # Estimate data pdf with KDE on a random subsample
    kde_skl = KernelDensity(kernel="gaussian", bandwidth=4.0).fit(modelx[:, np.newaxis])
    kde = np.exp(kde_skl.score_samples(x_grid[:, np.newaxis]))

    # Find cutoff
    kdethi = np.argmax(kde[::-1] > kde.max() * 0.5)

    # Fit X^2
    param = chi2.fit(modelx, coil_elements)
    chi_pdf = chi2.pdf(x_grid, *param[:-2], loc=param[-2], scale=param[-1])

    # Compute goodness-of-fit (gof)
    gof = float(np.abs(kde[-kdethi:] - chi_pdf[-kdethi:]).mean())
    if save_plot:
        out_file = plot_qi2(x_grid, kde, chi_pdf, modelx, kdethi)

    # return gof, out_file
    return gof

def make_airmask(bg_data, noNeck_data):
    air_data = (1 - bg_data) * noNeck_data
    std_signal_air = round(air_data[air_data!=0].std(),2)
    return air_data, std_signal_air

def make_artimg(air_data):
    X,Y,Z = air_data.shape
    art_img = np.zeros((X,Y,Z))
    air_data_tmp = air_data[air_data!=0].flatten()
    hist_amp, hist_value = np.histogram(air_data_tmp,335)
    noise_trh = hist_value[np.where(hist_amp == hist_amp.max())]

    for x in range(X):
        for y in range(Y):
            for z in range(Z):
                voxelvalue = air_data[x,y,z]
                if air_data[x,y,z] <= noise_trh:
                    art_img[x,y,z] = 0
                else:
                    art_img[x,y,z] = 1
    return art_img

def make_wm_gm_masks(seg_data):
    X, Y, Z = seg_data.shape
    wm_mask = np.zeros((X,Y,Z))
    gm_mask = np.zeros((X,Y,Z))
    for x in range(X):
        for y in range(Y):
            for z in range(Z):
                if seg_data[x,y,z] == 3:
                    wm_mask[x,y,z] = 1
                elif seg_data[x,y,z] == 2:
                    gm_mask[x,y,z] = 1
    return wm_mask, gm_mask

def read_brainregionsfile(brainregionsfile):
    with open(brainregionsfile) as brainregions_file:
        brainregions = brainregions_file.read()

    brainregions_col = brainregions.split('\n')
    columns = [c.split()  for c in brainregions_col]
    region_values = [c[1] for c in columns]
    region_names = [c[0] for c in columns]
    return region_values,region_names

def write_qmfile(filename,cnr,cjv,qi1,qi2):
    with open( output_folder + '/' + filename+'_qm.txt','w') as quality_metrics:
        quality_metrics.write('CNR: ' + str(cnr) + '\n' + 'CJV: ' + str(cjv) + '\n' + 'QI1: ' + str(qi1) + '\n' + 'QI2: ' + str(qi2))

def compare_list_lengths(lists):
    lengths = [len(lst) for lst in lists]
    if len(set(lengths)) != 1:
        print("Error: Not all lists have the same length!")
        sys.exit(1)
    else:
        print("All lists have the same length.")


#-------------

# Specify the folder containing output files
output_folder = 'MetaOutputsGUI'

# Check if the output folder exists, if not, create it
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"Output folder '{output_folder}' created.")


#aalatlas inladen
atlas_data = nibabel.load('AAL.nii').get_fdata()

region_values,region_names = read_brainregionsfile('brainregions.txt')
files_list = []
files_outskin = []
files_noNeck = []
files_seg = []
file_names = []

# Check if the folder exists
if os.path.exists(folder_path):
    # Iterate through all files in the folder
    for file_name in os.listdir(folder_path):
        
        # Check if the file ends with the postfixes needed and add them
        if file_name.endswith('_struct.nii.gz'):
            shutil.copy(os.path.join(folder_path, file_name), os.path.join(output_folder, file_name))
            files_list.append(os.path.join(folder_path, file_name))
            identifier, tag = file_name.split("_", 1)
            file_names.append(identifier)

        elif file_name.endswith('_outskin_mask.nii.gz'):
            files_outskin.append(os.path.join(folder_path, file_name))

        elif file_name.endswith('_noNeck.nii.gz'):
            files_noNeck.append(os.path.join(folder_path, file_name))

        elif file_name.endswith('_structSeg_seg.nii.gz'):
            files_seg.append(os.path.join(folder_path, file_name))

lists = [files_list,files_outskin,files_noNeck,files_seg,file_names]
compare_list_lengths(lists)

for i, file in enumerate(files_list):
    img_data = nibabel.load(file).get_fdata()
    affine_matr = nibabel.load(file).affine
    bg_data = nibabel.load(files_outskin[i]).get_fdata()
    noNeck_data = nibabel.load(files_noNeck[i]).get_fdata()
    seg_data = nibabel.load(files_seg[i]).get_fdata()

    air_img, std_air = make_airmask(bg_data, noNeck_data)
    wm_mask, gm_mask = make_wm_gm_masks(seg_data)
    art_img = make_artimg(air_img)

    make_snrmap(img_data,region_values,atlas_data,file_names[i],affine_matr,std_air, output_folder)

    # Check if the files with postfixes exist in the output folder
    files_exist = os.path.exists(os.path.join(output_folder, f"{file_names[i]}_qm.txt"))
    if files_exist:
        print("Quality metrics textfile already exist. Skipping generation.")

    else:    
        print("CJV: ", calculate_cjv(img_data,wm_mask,gm_mask))
        print("CNR: ", calculate_cnr(img_data,wm_mask,gm_mask,std_air)) 
        print('QI1: ', art_qi1(air_img,art_img))
        print('QI2: ', art_qi1(air_img,art_img)+art_qi2(noNeck_data,air_img))
        write_qmfile(file_names[i],calculate_cjv(img_data,wm_mask,gm_mask),calculate_cnr(img_data,wm_mask,gm_mask,std_air),art_qi1(air_img,art_img),art_qi1(air_img,art_img)+art_qi2(noNeck_data,air_img))
