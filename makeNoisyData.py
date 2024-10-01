import numpy as np
import nibabel as nib
from sklearn import preprocessing
import matplotlib.pyplot as plt
import nibabel
import raster_geometry as rg

# Read Image
img = nib.load('my_warped_atlas.nii.gz').get_fdata() # 3D image

# Read Image
img_int_tmp = nib.load('my_warped_structural.nii.gz')
img_int = img_int_tmp.get_fdata() # 3D image

# Binarize atlas
Binarized_mask = np.zeros((img.shape))
Binarized_Atlas = np.zeros((img.shape))
for slice in range(img.shape[0]):
    Binarized_mask_tmp = preprocessing.Binarizer(threshold=0).transform(img[slice,:,:])
    Binarized_mask[slice,:,:] = Binarized_mask_tmp

#     Binarized_Atlas_tmp = preprocessing.Binarizer(threshold=0).transform(img[slice,:,:])
#     Binarized_Atlas[slice,:,:] = Binarized_Atlas_tmp

# Binarized_Atlas[Binarized_Atlas == 1] = 128


# Convert the image to 2D
#img_2d = img[20,:,:]

# Generate noise with same shape as that of the image
#noise on the left half of the brain
noise_l = np.random.normal(0, 25, img_int.shape) 
noise_l[0:46,:,:] = 0

#noise on the anterior half of the brain
noise_a = np.random.normal(0, 25, img_int.shape) 
noise_a[:,0:55,:] = 0

#noise in the middle
arr = rg.ellipsoid((91, 109, 91), (20,30,15), (0.5,0.5,0.5))
noise_mid = np.random.normal(0, 25, img_int.shape) 
noise_mid = noise_mid * arr

#noise on the cortex
arr = rg.ellipsoid((91, 109, 91), (20,30,15), (0.5,0.5,0.5))
arr = 1 - arr
noise_cort = np.random.normal(0, 25, img_int.shape) 
noise_cort = noise_cort * arr

# Add the noise to the image
img_noised_l = img_int + noise_l
img_noised_a = img_int + noise_a
img_noised_mid = img_int + noise_mid
img_noised_cort = img_int + noise_cort

# Clip the pixel values to be between 0 and 255.
img_noised_l = np.clip(img_noised_l, 0, 255).astype(np.uint8)
img_noised_a = np.clip(img_noised_a, 0, 255).astype(np.uint8)
img_noised_mid = np.clip(img_noised_mid, 0, 255).astype(np.uint8)
img_noised_cort = np.clip(img_noised_cort, 0, 255).astype(np.uint8)


def snr_of_certain_region(img,mask):
    """ 
    Input image and mask with the same size and calculate the SNR in that area
    """
    foreground = np.multiply(img,mask)
    mu_fg = foreground[foreground!=0].mean()
    # print(mu_fg)
    sigma_fg = foreground[foreground!=0].std()
    # print(sigma_fg)
    num_fg = np.count_nonzero(foreground)
    # print(num_fg)
    return float(mu_fg / (sigma_fg * (num_fg / (num_fg - 1))**0.5))

# for mu in range(0,31,10):
#     for sigma in range(0,81,10):
#         noise = np.random.normal(mu, sigma, img_int.shape) 
#         img_noised = img_int + noise
#         #img_noised = np.clip(img_noised, 0, 255).astype(np.uint8)
#         #print(snr_of_certain_region(img_noised,Binarized_mask))

# img_noisedconv_l = np.array(img_noised_l, dtype=np.float32)
# img_noised_l = nibabel.Nifti1Image(img_noisedconv_l, affine=img_int_tmp.affine)
# nibabel.save(img_noised_l, 'img_noised_l')  

# img_noisedconv_a = np.array(img_noised_a, dtype=np.float32)
# img_noised_a = nibabel.Nifti1Image(img_noisedconv_a, affine=img_int_tmp.affine)
# nibabel.save(img_noised_a, 'img_noised_a')  

# img_noisedconv_mid = np.array(img_noised_mid, dtype=np.float32)
# img_noised_mid = nibabel.Nifti1Image(img_noisedconv_mid, affine=img_int_tmp.affine)
# nibabel.save(img_noised_mid, 'img_noised_mid')  

# img_noisedconv_cort = np.array(img_noised_cort, dtype=np.float32)
# img_noised_cort = nibabel.Nifti1Image(img_noisedconv_cort, affine=img_int_tmp.affine)
# nibabel.save(img_noised_cort, 'img_noised_cort')  


#noise on the inferior half of the brain
noise_i = np.random.normal(0, 25, img_int.shape) 
noise_i[:,:,0:46] = 0
img_noised_i = img_int + noise_i
img_noised_i = np.clip(img_noised_i, 0, 255).astype(np.uint8)

img_noisedconv_i = np.array(img_noised_i, dtype=np.float32)
img_noised_i = nibabel.Nifti1Image(img_noisedconv_i, affine=img_int_tmp.affine)
nibabel.save(img_noised_i, 'img_noised_i')  

#signal+noise on the inferior half of the brain
noise_i2 = np.random.normal(5, 25, img_int.shape) 
noise_i2[:,:,0:46] = 0
img_noised_i2 = img_int + noise_i2
img_noised_i2 = np.clip(img_noised_i2, 0, 255).astype(np.uint8)

img_noisedconv_i2 = np.array(img_noised_i2, dtype=np.float32)
img_noised_i2 = nibabel.Nifti1Image(img_noisedconv_i2, affine=img_int_tmp.affine)
nibabel.save(img_noised_i2, 'img_noised_i2')  

#signal+noise on the inferior half of the brain
noise_i3 = np.random.normal(10, 25, img_int.shape) 
noise_i3[:,:,0:46] = 0
img_noised_i3 = img_int + noise_i3
img_noised_i3 = np.clip(img_noised_i3, 0, 255).astype(np.uint8)

img_noisedconv_i3 = np.array(img_noised_i3, dtype=np.float32)
img_noised_i3 = nibabel.Nifti1Image(img_noisedconv_i3, affine=img_int_tmp.affine)
nibabel.save(img_noised_i3, 'img_noised_i3')  