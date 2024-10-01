import numpy as np
import nibabel as nib
from plantcv import plantcv as pcv
import matplotlib.pyplot as plt
from skimage.transform import resize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap
from ipywidgets import interact
import colorspacious as cs
import cv2

target_data = nib.load('my_warped_structural.nii.gz') # 3D image
target_img = target_data.get_fdata() # 3D image

warpfield_data = nib.load('my_warpfield.nii.gz') # 4D image
warpfield = warpfield_data.get_fdata() # 4D image

X, Y, Z, A = warpfield.shape
sum_def = 0
print(warpfield.shape)

newwarpfield = np.zeros((X,Y,Z))
for x in range(X):
    for y in range(Y):
        for z in range(Z):
            for a in range(A):
                sum_def += abs(warpfield[x,y,z,a])
                newwarpfield[x,y,z] += abs(warpfield[x,y,z,a])
print(sum_def)

aff_matr = np.identity(4)
newwarpfield_tmp = np.array(newwarpfield, dtype=np.float32)
newwarpfield_tmp = nib.Nifti1Image(newwarpfield_tmp, affine=aff_matr)
nib.save(newwarpfield_tmp, 'newwarpfield')

affwarpfield = np.array(warpfield, dtype=np.float32)
affwarpfield = nib.Nifti1Image(affwarpfield, affine=aff_matr)
nib.save(affwarpfield, 'affwarpfield')

newwarpfield_a = np.array(newwarpfield, dtype=np.float32)
newwarpfield_a = nib.Nifti1Image(newwarpfield_a, affine=aff_matr)
nib.save(newwarpfield_a, 'newwarpfield_a')

newwarpfield = np.interp(newwarpfield, (newwarpfield.min(), newwarpfield.max()), (0, +1))
target_img = np.interp(target_img, (target_img.min(), target_img.max()), (0, +1))


print(newwarpfield.shape)
grey_image = newwarpfield
# # Apply the Inferno colormap to the grayscale warpfield
# inferno_image = plt.cm.inferno(grey_image)

# Get the color map by name:
cm = plt.get_cmap('inferno')

# Apply the colormap like a function to any array:
inferno_image = cm(grey_image)
print(inferno_image.shape)


def grayscale_to_cmyk(grey_image):
    # Ensure the input image is in the range [0, 1]
    grey_image = np.clip(grey_image, 0, 1)

    # Create CMYK channels
    cmyk_image = np.zeros((*grey_image.shape, 4))
    
    # CMY channels remain zero
    cmyk_image[..., 0:3] = 0
    
    # K (black) channel is the grayscale intensity
    cmyk_image[..., 3] = 1 - grey_image

    return cmyk_image


grey_image = target_img
# Convert grayscale image to CMYK but in grayscale
cmyk_image = grayscale_to_cmyk(grey_image)

alpha = 0.6
out_img = (alpha * inferno_image[:, :, :, :]) + ((1 - alpha) * cmyk_image[:, :, :, :])
out_img2 = np.array(out_img, dtype=np.float32)
out_img2 = nib.Nifti1Image(out_img2, affine=aff_matr)
nib.save(out_img2, 'out_img2')

inferno_image2 = np.copy(inferno_image)
inferno_image2[:, :, :, 3] = 0.6
out_img7 = inferno_image2 + ((1 - alpha) * cmyk_image[:, :, :, :])
out_img7_tmp = np.array(out_img7, dtype=np.float32)
out_img7_tmp = nib.Nifti1Image(out_img7_tmp, affine=aff_matr)
nib.save(out_img7_tmp, 'out_img7')

# Scale the values to [0, 255] (assuming values are in [0, 1])
gray_matrix_scaled = grey_image

# Create an empty RGB image with the same dimensions as the grayscale image
height, width, depth = grey_image.shape
rgb_image = np.zeros((height, width, depth, 4))

# Assign the grayscale values to all three channels in the RGB image
rgb_image[:, :, :, 0] = gray_matrix_scaled  # Red channel
rgb_image[:, :, :, 1] = gray_matrix_scaled  # Green channel
rgb_image[:, :, :, 2] = gray_matrix_scaled  # Blue channel
rgb_image[:, :, :, 3] = 1.0  # Opactity channel


rgb_image2 = np.copy(rgb_image)
rgb_image2[:,:,:,3] = 0.4
inferno_image2 = np.copy(inferno_image)
inferno_image2[:,:,:,3] = 0.6


out_img3 = inferno_image2 + rgb_image2
out_img3_tmp = np.array(out_img3, dtype=np.float32)
out_img3_tmp = nib.Nifti1Image(out_img3_tmp, affine=aff_matr)
nib.save(out_img3_tmp, 'out_img3')

out_img4_tmp = np.array(out_img3[:,:,:,0], dtype=np.float32)
out_img4_tmp = nib.Nifti1Image(out_img4_tmp, affine=aff_matr)
nib.save(out_img4_tmp, 'out_imgR')

out_img5_tmp = np.array(out_img3[:,:,:,1], dtype=np.float32)
out_img5_tmp = nib.Nifti1Image(out_img5_tmp, affine=aff_matr)
nib.save(out_img5_tmp, 'out_imgG')

out_img6_tmp = np.array(out_img3[:,:,:,2], dtype=np.float32)
out_img6_tmp = nib.Nifti1Image(out_img6_tmp, affine=aff_matr)
nib.save(out_img6_tmp, 'out_imgB')

# plt.subplot(1, 6, 1)
# plt.imshow(inferno_image[:, :, 55])
# plt.subplot(1, 6, 2)
# plt.imshow(cmyk_image[:, :, 55])
# plt.subplot(1, 6, 3)
# plt.imshow(out_img[:, :, 55])
# plt.subplot(1, 6, 4)
# plt.imshow(rgb_image[:, :, 55])
# plt.subplot(1, 6, 5)
# plt.imshow(out_img3[:, :, 55])
# plt.subplot(1, 6, 6)
# plt.imshow(out_img7[:, :, 55])
# plt.show()




noise1 = np.random.normal(0, 25, newwarpfield.shape) 
noise2 = np.random.normal(0, 25, newwarpfield.shape) 
noise3 = np.random.normal(0, 25, newwarpfield.shape)
noise4 = np.random.normal(0, 25, newwarpfield.shape)
noise5 = np.random.normal(0, 25, newwarpfield.shape)

newwarpfield1 = newwarpfield + noise1
newwarpfield2 = newwarpfield + noise2
newwarpfield3 = newwarpfield + noise3
newwarpfield4 = newwarpfield + noise4
newwarpfield5 = newwarpfield + noise5

warpfields = [newwarpfield1,newwarpfield2,newwarpfield3,newwarpfield4,newwarpfield5]

avg_warpfield = np.mean(warpfields, axis=0)
std_warpfield = np.std(warpfields, axis=0)
z_warpfield1 = (newwarpfield1 - avg_warpfield)/std_warpfield
print(avg_warpfield.shape)
print(std_warpfield.shape)
print(z_warpfield1.shape)
