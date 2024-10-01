import raster_geometry as rg
import matplotlib.pyplot as plt
from skimage import measure
import numpy as np
import nibabel as nib
from numpy import loadtxt

# Read Image
img = nib.load('my_warped_atlas.nii.gz').get_fdata() # 3D image
target_data = nib.load('my_warped_structural.nii.gz') # 3D image
target_img = target_data.get_fdata() # 3D image
warpfield = nib.load('my_warpfield.nii.gz').get_fdata() # 4D image
newwarpfield = nib.load('my_field.nii.gz').get_fdata() # 4D image
warpcoeff = nib.load('my_nonlinear_transf.nii.gz').get_fdata() # 4D image

warpcoeff1 = nib.load(r'C:/Users/maaik/OneDrive/Documents/UU/MinorResearchProject/Stefan/shellCommandTest/RealWrapperData/ellipsoid/my_nonlinear_transfMNI1.nii.gz').get_fdata() # 4D image
warpcoeff2 = nib.load(r'C:/Users/maaik/OneDrive/Documents/UU/MinorResearchProject/Stefan/shellCommandTest/RealWrapperData/ellipsoid/my_nonlinear_transfMNI2.nii.gz').get_fdata() # 4D image
warpcoeff3 = nib.load(r'C:/Users/maaik/OneDrive/Documents/UU/MinorResearchProject/Stefan/shellCommandTest/RealWrapperData/ellipsoid/my_nonlinear_transfMNI3.nii.gz').get_fdata() # 4D image


arr = rg.ellipsoid((91, 109, 91), (36,41,30), (0.5,0.5,0.5))
aff_matr = np.identity(4)
ellipsoid1_tmp = np.array(arr, dtype=np.float32)
ellipsoid1 = nib.Nifti1Image(ellipsoid1_tmp, affine=target_data.affine)
nib.save(ellipsoid1, 'ellipsoid7v2')  

# arr2 = rg.ellipsoid((91, 109, 91), (35,51,32), (0.5,0.5,0.5))
# ellipsoid2_tmp = np.array(arr2, dtype=np.float32)
# ellipsoid2 = nib.Nifti1Image(ellipsoid2_tmp, affine=target_data.affine)
# nib.save(ellipsoid2, 'ellipsoid2v2')  

# arr3 = rg.ellipsoid((91, 109, 91), (35,49,32), (0.5,0.5,0.5))
# ellipsoid3_tmp = np.array(arr3, dtype=np.float32)
# ellipsoid3 = nib.Nifti1Image(ellipsoid3_tmp, affine=target_data.affine)
# nib.save(ellipsoid3, 'ellipsoid3v2')

# arr4 = rg.ellipsoid((91, 109, 91), (35,48,32), (0.5,0.5,0.5))
# ellipsoid4_tmp = np.array(arr4, dtype=np.float32)
# ellipsoid4 = nib.Nifti1Image(ellipsoid4_tmp, affine=target_data.affine)
# nib.save(ellipsoid4, 'ellipsoid4v2')  

# arr5 = rg.ellipsoid((91, 109, 91), (35,46,32), (0.5,0.5,0.5))
# ellipsoid5_tmp = np.array(arr5, dtype=np.float32)
# ellipsoid5 = nib.Nifti1Image(ellipsoid5_tmp, affine=target_data.affine)
# nib.save(ellipsoid5, 'ellipsoid5v2')  

# arr6 = rg.ellipsoid((91, 109, 91), (35,45,32), (0.5,0.5,0.5))
# ellipsoid6_tmp = np.array(arr6, dtype=np.float32)
# ellipsoid6 = nib.Nifti1Image(ellipsoid6_tmp, affine=target_data.affine)
# nib.save(ellipsoid6, 'ellipsoid6v2')


### AFFINE & NON-LINEAR TRANSFORMATION

# #import text file into NumPy array as integer
# aff_matr = loadtxt('my_affine_transf.txt', dtype='int')
# #display content of text file
# print(aff_matr)

# print(img.shape)
# X, Y, Z = img.shape
# new_img = np.zeros((img.shape))
# for x in range(2):
#     for y in range(2):
#         for z in range(2):
#             old_coord = [x,y,z,1]
#             new_coord = aff_matr @ old_coord
#             print(old_coord)
#             print(new_coord)
#             new_img[new_coord] = img[x,y,z]


# warp_matr = [warpfield[:,:,:,0],warpfield[:,:,:,1],warpfield[:,:,:,2],0]
# warp_img = np.zeros((img.shape))
# for x in range(2):
#     for y in range(2):
#         for z in range(2):
#             old_coord = [x,y,z,1]
#             new_coord = old_coord + warp_matr

#def_img = aff_matr @ img

# plt.clf()
# plt.subplot(1, 3, 1)
# plt.imshow(img[10,:,:])
# plt.subplot(1, 3, 2)
# plt.imshow(def_img[10,:,:,1])
# plt.subplot(1, 3, 3)
# plt.imshow(target_img[10,:,:])
# plt.show()

X, Y, Z, A = warpcoeff.shape
sum_def = 0
sum_def1 = 0
sum_def2 = 0
sum_def3 = 0
for x in range(X):
    for y in range(Y):
        for z in range(Z):
            for a in range(A):
                sum_def += abs(warpcoeff[x,y,z,a])
                sum_def1 += abs(warpcoeff1[x,y,z,a])
                sum_def2 += abs(warpcoeff2[x,y,z,a])
                sum_def3 += abs(warpcoeff3[x,y,z,a])
print(sum_def)
print(sum_def1)
print(sum_def2)
print(sum_def3)


print(warpcoeff.shape)
print(warpcoeff.max())
print(warpcoeff.min())

trf1 = np.zeros(warpcoeff.shape)
trf2 = np.zeros(warpcoeff.shape)
trf3 = np.zeros(warpcoeff.shape)
trf4 = np.zeros(warpcoeff.shape)
trf5 = np.zeros(warpcoeff.shape)
trf6 = np.zeros(warpcoeff.shape)

trf2[:,:,:,:] = warpcoeff.max()
trf3[:,:,:,:] = warpcoeff.min()
trf4[:,0:12,:,0] = warpcoeff.min()
trf4[:,12:24,:,0] = warpcoeff.max()
trf5[:,0:12,:,1] = warpcoeff.min()
trf5[:,12:24,:,1] = warpcoeff.max()
trf6[:,0:12,:,2] = warpcoeff.min()
trf6[:,12:24,:,2] = warpcoeff.max()

list = [trf1,trf2,trf3,trf4,trf5,trf6]
i = 1
for item in list:
    trf = np.array(item, dtype=np.float32)
    trf = nib.Nifti1Image(trf, affine=target_data.affine)
    nib.save(trf, 'trf' + str(i))
    i += 1


# warpcoeff[:,0:12,:,1] = -10
# warpcoeff[:,12:24,:,1] = 15
# warp1 = np.array(warpcoeff, dtype=np.float32)
# warp1 = nib.Nifti1Image(warp1, affine=target_data.affine)
# nib.save(warp1, 'warp1')


# print(warpfield.shape)
# print(warpfield.max())
# print(warpfield.min())

# print(newwarpfield.shape)
# print(newwarpfield.max())
# print(newwarpfield.min())


# test1 = np.zeros(warpfield.shape)
# test2 = np.zeros(warpfield.shape)
# test3 = np.zeros(warpfield.shape)
# test4 = np.zeros(warpfield.shape)
# test5 = np.zeros(warpfield.shape)
# test6 = np.zeros(warpfield.shape)
# test7 = np.zeros(warpfield.shape)

# test2[:,:,:,:] = warpfield.max()
# test3[:,:,:,:] = warpfield.min()
# test4[:,0:55,:,0] = warpfield.min()
# # test4[:,55:109,:,0] = warpfield.max()
# test5[:,0:55,:,0] = warpfield.max()
# # test5[:,55:109,:,1] = warpfield.max()
# # test6[:,0:55,:,0] = warpfield.max()
# test6[:,55:109,:,0] = warpfield.min()
# test7[:,55:109,:,0] = warpfield.max()

# list = [test1,test2,test3,test4,test5,test6,test7]
# i = 1
# for item in list:
#     test = np.array(item, dtype=np.float32)
#     test = nib.Nifti1Image(trf, affine=target_data.affine)
#     nib.save(test, 'test' + str(i))
#     i += 1