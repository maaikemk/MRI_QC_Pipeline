import nibabel as nib
from scipy.ndimage import affine_transform
import numpy as np

img = nib.load('SynthData/myBrainOrig.nii.gz')

def make_synth_data(img):
    data = img.get_fdata()
    affine = img.affine

    # Define the range of transformations (e.g., rotation angles, shearing factors)
    rotation_angles = np.linspace(0, 30, 20)  # Specify your rotation angles
    shearing_factors = np.linspace(0, 0.1, 20)  # Specify your shearing factors

    transformed_images = []

    for angle, shear in zip(rotation_angles, shearing_factors):
        # Generate transformation matrix
        rotation_matrix = np.array([[1, 0, 0],
                                    [0, np.cos(np.radians(angle)), -np.sin(np.radians(angle))],
                                    [0, np.sin(np.radians(angle)), np.cos(np.radians(angle))]])
        shear_matrix = np.array([[1, 0, 0],
                                [0, 1, shear],
                                [0, 0, 1]])

        # Combine transformations
        transformation_matrix = np.dot(rotation_matrix, shear_matrix)

        # Apply affine transformation to the data
        transformed_data = affine_transform(data, transformation_matrix, offset=0, order=3, mode='constant', cval=0.0, prefilter=False)

        # Create a new NIfTI image with the transformed data and original affine
        transformed_img = nib.Nifti1Image(transformed_data, affine)
        transformed_images.append(transformed_img)

        for idx, t_img in enumerate(transformed_images):
            nib.save(t_img, f'SynthData/transformedImage{idx}.nii')

make_synth_data(img)