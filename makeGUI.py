import tkinter as tk
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import os
from collections import defaultdict
from tkinter import filedialog  # Import file dialog from tkinter
import shutil
import matplotlib.colors
import glob
import re

# Define the directory path of the images and of the txt files (initially set to None)
folder_paths = []

reference_set = "Reference_Set"

def select_folder_paths():
    """
    Select multiple folder paths using a file dialog and store them in the global variable folder_paths.

    Returns:
        None
    """
    global folder_paths
    while True:
        folder_path = filedialog.askdirectory()  # Open file dialog for selecting a folder
        if folder_path:
            folder_paths.append(folder_path)
            print("Selected folder path:", folder_path)
        else:
            break

def analyse_images_and_close_window():
    """
    Closes the Tkinter first window.

    Returns:
        None
    """
    # Close the window/
    root.destroy()

# Create a root window
root = tk.Tk()

# Create buttons to open file dialogs for selecting folders
select_folder_button = tk.Button(root, text="Select Images Folders", command=select_folder_paths)
select_folder_button.pack()

# Create a button to analyze selected images and close the window
analyse_button = tk.Button(root, text="Analyse Selected Images", command=analyse_images_and_close_window)
analyse_button.pack()

# Run the root window
root.mainloop()

# Create a new folder to combine all files
combined_folder = "New_Scans"
if os.path.exists(combined_folder):
    folder_exists = True
else:
    folder_exists = False
    os.makedirs(combined_folder, exist_ok=True)

    # Copy all files from selected folders to the new combined folder
    for folder_path in folder_paths:
        for filename in os.listdir(folder_path):
            shutil.copy(os.path.join(folder_path, filename), os.path.join(combined_folder, filename))

# Now use the combined_folder in the rest of the code
folder_path = combined_folder  
print(folder_path)

# Create a defaultdict to store files by identifier
files_by_identifier = defaultdict(list)
files_txt = []
snrlist_txt = []
snrDlist_txt = []
brainmask_nii = []

# Define the order of tags and corresponding index
tag_order = {"signal.nii": 0, "struct.nii.gz": 1,  "std.nii": 2, "snr.nii": 3, "snrD.nii": 4, "field.nii": 5, "fieldZmap.nii": 6, "snrZmap.nii": 7, "snrDZmap.nii": 8}  # Example order and indices

# List all files in the directory
files = os.listdir(folder_path)

# Iterate through each file
for file_name in files:
    # print(file_name)
    # Split the file name into identifier and tag
    identifier, tag = file_name.split("_", 1)
    if tag == 'qm.txt':
        files_txt.append(folder_path + '/'+ file_name)
    elif tag == 'snrZmap.txt':
        snrlist_txt.append(file_name)
    elif tag == 'snrDZmap.txt':
        snrDlist_txt.append(file_name)
    elif tag == 'brainmask.nii':
        brainmask_nii.append(file_name)
    else:
        # Add the file to the corresponding identifier list
        files_by_identifier[identifier].append((file_name, tag_order.get(tag, len(tag_order))))  # Store file and index

# Sort files within each identifier group based on indices
for identifier, file_list in files_by_identifier.items():
    sorted_files = sorted(file_list, key=lambda x: x[1])  # Sort based on index
    files_by_identifier[identifier] = sorted_files

# Initialize a list to store the grouped and sorted files
grouped_sorted_lists = []

# Populate the grouped_sorted_lists with sorted files
for identifier, sorted_files in files_by_identifier.items():
    file_list = [file_name for file_name, _ in sorted_files]  # Extract file names
    grouped_sorted_lists.append(file_list)

nifti_data = grouped_sorted_lists
print('Made lists')

struct_list = []
id_idx = []
# for list in nifti_data:
#     struct_list.append(list[1])
#     # Iterate through the struct files 
#     for i in range(len(struct_list)):
#         for j in range(i + 1, len(struct_list)):
#             nifti_i = nib.load(folder_path+'/'+struct_list[i]).get_fdata()
#             nifti_j = nib.load(folder_path+'/'+struct_list[j]).get_fdata()

#             if np.array_equal(nifti_i, nifti_j):
#                 print(f"Warning: {struct_list[i]} and {struct_list[j]} are identical! Removing the latter")
#                 id_idx.append(j)

#         for index in sorted(id_idx, reverse=True):
#             del nifti_data[index]
#             del files_txt[index]
#             del snrlist_txt[index]
#             del snrDlist_txt[index]

print('Sorting done')

def qm_files_reference(reference_set):
    files_txt_ref = []
    snrlist_txt_ref = []
    snrDlist_txt_ref = []
    # List all files in the directory
    files = os.listdir(reference_set)

    # Iterate through each file
    for file_name in files:
        # print(file_name)
        # Split the file name into identifier and tag
        identifier, tag = file_name.split("_", 1)
        if tag == 'qm.txt':
            files_txt_ref.append(reference_set + '/' + file_name)
        elif tag == 'snrZmap.txt':
            snrlist_txt_ref.append(reference_set + '/' + file_name)
        elif tag == 'snrDZmap.txt':
            snrDlist_txt_ref.append(reference_set + '/' + file_name)
    return files_txt_ref, snrlist_txt_ref, snrDlist_txt_ref

files_txt_ref, snrlist_txt_ref, snrDlist_txt_ref = qm_files_reference(reference_set)


###################

def calculate_Zmap(list_txt):  
    """
    Make a list of list of all values per brain region of all txt files. Calculate 
    the average and standard deviation per brain region.

    list_txt: list of str, list of filenames containing data in the format of 'brain 
    region name: brain region value \n'

    Returns tuple of 
    avg_reg: 1D numpy array, containing average values per brain region
    std_reg: 1D numpy array, containing standard deviations per brain region
    region_arrays: list of lists, arrays containing region data of all files in the dataset
    """
    region_arrays = [[]]
    region_values = []
    for i, filename in enumerate(list_txt):  
        with open(filename, 'r') as file:
            lines = file.readlines()                
            for j, line in enumerate(lines):
                entries = line.strip().split()
                if len(entries) >= 2:
                    if len(region_arrays) < i+1:
                        region_arrays.append([])
                    value = float(entries[1])
                    region_arrays[i].append(value)
                    if i == 0: 
                        # Collect brain region labels one time
                        region_values.append(entries[0])
    region_arrays = np.array(region_arrays, dtype=np.float64)                
    region_arrays[np.isinf(region_arrays)] = np.nan
    avg_reg = np.nanmean(region_arrays, axis=0)
    std_reg = np.nanstd(region_arrays, axis=0)

    return avg_reg, std_reg, region_arrays, region_values

def create_Zmap(atlas_img, total_zvalues_file, snrlist_txt):
    """
    Create a matrix of by calculating z-values per brain region of an image.

    atlas_img: numpy array, input image data containing brain region labels
    avg_reg: numpy array, average of regions
    std_reg: numpy array, standard deviation of regions
    region_arrays: list of lists, arrays containing region data of all files in the dataset

    Returns:
    z_matrix: numpy array, the generated matrix containing z-values per brain region
    """
    avg_reg = []
    std_reg = []
    region_values = []
    region_arrays = []

    with open(reference_set+ '/' + total_zvalues_file, 'r') as file:
        for line in file:
            items = line.strip().split()
            avg_reg.append(float(items[0]))
            std_reg.append(float(items[1]))
            region_values.append(float(items[2]))

    with open(folder_path+'/'+snrlist_txt, 'r') as file:
        for line in file:
            entries = line.strip().split()
            if len(entries) >= 2:
                value = float(entries[1])
                region_arrays.append(value)

    X, Y, Z = atlas_img.shape
    z_matrix = np.zeros((X,Y,Z))
    for value in range(len(region_values)):
        gray_value = int(region_values[value])
        voxelcoordx = []
        voxelcoordy = []
        voxelcoordz = []
        for x in range(X):
            for y in range(Y):
                for z in range(Z):
                    voxelvalue = atlas_img[x,y,z]
                    if voxelvalue == gray_value:
                        voxelcoordx.append(x)
                        voxelcoordy.append(y)
                        voxelcoordz.append(z)     

        for coord in range(len(voxelcoordx)):
            z_matrix[voxelcoordx[coord],voxelcoordy[coord],voxelcoordz[coord]] = (region_arrays[value] - avg_reg[value])/std_reg[value]
    return z_matrix

def calculate_z_def_fields(output_filename):
    """
    Calculate the average and standard deviation per voxel of the deformation fields NIfTI files
    which can be used for z-field calculations.

    Returns:
    None
    """
    warpfields = []

    # List all files in the directory
    all_files = os.listdir(reference_set)
    print(all_files)

    for file_name in all_files:
        if file_name.endswith('_total_fieldZmap.nii'):
            namelist = file_name.split("_")
            identifier = namelist[1]
            new_path = os.path.join(reference_set, identifier+'_fieldZmap.nii')
            shutil.move(os.path.join(reference_set, file_name), new_path)

    all_files = os.listdir(reference_set)
    print(all_files)
    # Filter files ending with '_fieldZmap.nii'
    nifti_files = [f for f in all_files if f.endswith('_fieldZmap.nii')]

    # Load each NIfTI file and add the data to the list
    for nifti_file in nifti_files:
        file_path = os.path.join(reference_set, nifti_file)
        nifti_img = nib.load(file_path)
        nifti_data = nifti_img.get_fdata()
        warpfields.append(nifti_data)

    avg_warpfield = np.mean(warpfields, axis=0)
    std_warpfield = np.std(warpfields, axis=0)

    with open(reference_set + '/' + output_filename,'w') as f:
        # Write the first 3D matrix
        for i, matrix_2d in enumerate(avg_warpfield):
            f.write(f'# Matrix 1 - Slice {i}\n')  # Add a comment to indicate the slice index
            np.savetxt(f, matrix_2d, fmt='%.6f')
            f.write('\n')  # Add a newline to separate matrices

        # Write the second 3D matrix
        for i, matrix_2d in enumerate(std_warpfield):
            f.write(f'# Matrix 2 - Slice {i}\n')  # Add a comment to indicate the slice index
            np.savetxt(f, matrix_2d, fmt='%.6f')
            f.write('\n')  # Add a newline to separate matrices

def create_z_def_fields(test_data,identifier,affine_matrice,brainmask,total_zvalues_file):
    """
    Calculate the average and standard deviation per voxel of the deformation fields NIfTI files
    which can be used for z-field calculations.

    Returns:
    None
    """
    with open(reference_set+ '/' + total_zvalues_file, 'r') as f:
        content = f.read()

    slices = content.split('# Matrix ')
    matrix_3d_1_slices = [s for s in slices if s.startswith('1')]
    matrix_3d_2_slices = [s for s in slices if s.startswith('2')]

    avg_warpfield = []
    for slice_ in matrix_3d_1_slices:
        lines = slice_.strip().split('\n')
        matrix_2d = np.loadtxt(lines[1:])  # Skip the first line (slice index)
        avg_warpfield.append(matrix_2d)

    std_warpfield = []
    for slice_ in matrix_3d_2_slices:
        lines = slice_.strip().split('\n')
        matrix_2d = np.loadtxt(lines[1:])  # Skip the first line (slice index)
        std_warpfield.append(matrix_2d)

    avg_warpfield = np.array(avg_warpfield)
    std_warpfield = np.array(std_warpfield)

    # brainmask = nib.load(folder_path+'/'+brainmask_nii).get_fdata()
    z_field_pre = (test_data - avg_warpfield) / std_warpfield
    z_field = z_field_pre * brainmask
    nifti_array = np.array(z_field, dtype=np.float32)
    nifti_array = nib.Nifti1Image(nifti_array, affine=affine_matrice)
    nib.save(nifti_array, folder_path + '/' + identifier + '_fieldZmap.nii')

    nifti_array = np.array(z_field_pre, dtype=np.float32)
    nifti_array = nib.Nifti1Image(nifti_array, affine=affine_matrice)
    nib.save(nifti_array, folder_path + '/' + identifier + '_total_fieldZmap.nii')
    
def save_z_maps(avg_list,std_list,region_values,file_name):
    with open(reference_set + '/' + file_name,'w') as Zmap:
        for idx in range(len(region_values)):
            Zmap.write(str(avg_list[idx]) + ' ' + str(std_list[idx]) + ' ' + str(region_values[idx]) + '\n')

def remove_identical_struct_nifti(directory):
    tag = "_struct.nii.gz"
    nifti_files = glob.glob(os.path.join(directory, f"*{tag}"))
    identifiers_to_remove = set()

    def read_nifti_image(file_path):
        img = nib.load(file_path)
        return img.get_fdata()

    def compare_images(image1, image2):
        return np.array_equal(image1, image2)

    def extract_identifier(file_name, tag):
        return file_name.split(tag)[0]

    def remove_files_with_identifier(identifier):
        files_to_remove = glob.glob(os.path.join(directory, f"{identifier}*"))
        for file_path in files_to_remove:
            os.remove(file_path)
            print(f"Removed: {file_path}")

    # Compare all NIfTI files to find identical ones
    for i in range(len(nifti_files)):
        if nifti_files[i] in identifiers_to_remove:
            continue
        img1 = read_nifti_image(nifti_files[i])
        identifier_i = extract_identifier(os.path.basename(nifti_files[i]), tag)
        for j in range(i + 1, len(nifti_files)):
            if nifti_files[j] in identifiers_to_remove:
                continue
            img2 = read_nifti_image(nifti_files[j])
            if compare_images(img1, img2):
                identifier_j = extract_identifier(os.path.basename(nifti_files[j]), tag)
                identifiers_to_remove.add(nifti_files[j])
                remove_files_with_identifier(identifier_j)

    # Remove the identified files
    for file_to_remove in identifiers_to_remove:
        os.remove(file_to_remove)
        print(f"Removed: {file_to_remove}")



#############


if not folder_exists:
    print('Making z-score maps')    
    # warpfields = []
    # identifiers = []
    # affine_matrices = []
    # z_field_pre = []
    for i, dataset in enumerate(nifti_data):
        for file in dataset:
            if file.endswith('_snrZmap.nii'):
                nifti = nib.load(folder_path+'/'+file)
                test_data = nifti.get_fdata()            
                # z_matrix = create_Zmap(test_data, avg_reg, std_reg, region_arrays, region_values,i)
                z_matrix = create_Zmap(test_data, 'total_zmap_SNR.txt', snrlist_txt[i])

                identifier, tag = file.split("_", 1)
                nifti_array = np.array(z_matrix, dtype=np.float32)
                nifti_array = nib.Nifti1Image(nifti_array, affine=nifti.affine)
                nib.save(nifti_array, folder_path + '/' + identifier + '_snrZmap.nii') 

            elif file.endswith('_snrDZmap.nii'):
                nifti = nib.load(folder_path+'/'+file)
                test_data = nifti.get_fdata()            
                # z_matrix = create_Zmap(test_data, avg_reg2, std_reg2, region_arrays2, region_values2,i)
                z_matrix = create_Zmap(test_data, 'total_zmap_SNRD.txt', snrDlist_txt[i])

                identifier, tag = file.split("_", 1)
                nifti_array = np.array(z_matrix, dtype=np.float32)
                nifti_array = nib.Nifti1Image(nifti_array, affine=nifti.affine)
                nib.save(nifti_array, folder_path + '/' + identifier + '_snrDZmap.nii') 
           
            elif file.endswith('_fieldZmap.nii'):
                nifti = nib.load(folder_path+'/'+file)
                test_data = nifti.get_fdata() 
                affine_matrice = nifti.affine
                identifier, tag = file.split("_", 1)
                nifti_brainmask = nib.load(folder_path+'/'+identifier+'_brainmask.nii')
                brainmask = nifti_brainmask.get_fdata() 
                create_z_def_fields(test_data,identifier,affine_matrice,brainmask,'total_zmap_fields.txt')
                

##################

#Start second GUI
class NiftiViewer:
    def __init__(self, master, nifti_files_lists,files_txt,snrlist_txt,snrDlist_txt,files_txt_ref):
        self.master = master
        self.nifti_files_lists = nifti_files_lists
        self.default_nifti_files = nifti_files_lists[0]
        self.selected_nifti_files = self.default_nifti_files.copy()
        self.img_data = [None, None, None, None]
        self.current_slices = [0, 0, 0, 0]
        self.colormaps = [plt.cm.inferno, plt.cm.inferno, plt.cm.inferno, plt.cm.inferno]  # List of colormaps
        self.color_limits = [(0, 1), (0, 1), (0, 1), (0, 1)]  # Default color scale limits for each image
        self.sag_slice = False
        self.field = False
        self.fieldz = False
        self.snrz = False
        self.snrdz = False
        self.list_index = 0
        self.copy_called = False  

        # Create warning label
        self.warning_label = tk.Label(master, text="", fg="red")
        self.warning_label.grid(row=3, column=5, columnspan=1, padx=5, pady=5, sticky="W")  # Adjusted padding and grid position

        # Dictionary to store 1D arrays for each line number
        self.line_arrays = {}
        self.line_labels = {}
        self.curr_line_arrays = []
        self.curr_line_labels = []
        # Load the qm text files and populate the line arrays
        files_txt_total = files_txt + files_txt_ref
        for filename in files_txt_total:
            self.analyse_files_txt(filename)
                
        # Create a standalone canvas at row 4, column 2
        self.sum_canvas = tk.Canvas(master, width=100, height=300)
        self.sum_canvas.grid(row=4, column=2, padx=5, pady=5)  # Adjusted padding and grid position
       
        # Create a standalone canvas at row 3, column 1
        self.div_canvas = tk.Canvas(master, width=60, height=100)
        self.div_canvas.grid(row=3, column=1, padx=5, pady=5)  # Adjusted padding and grid position
        
        # Load NIfTI files
        self.load_nifti()

        # Create canvases to display images in a 2x2 grid
        self.canvases = []
        for i in range(4):
            row = (i // 2) * 3 + 1  # Calculate row index
            col = (i % 2) * 3  # Calculate column index

            # Get the shape of the current image
            img_shape = self.img_data[i].shape
            new_sizeratio = min(400 / img_shape[1], 300 / img_shape[0])
            
            # Calculate canvas width and height based on image dimensions
            canvas_width = img_shape[1]  
            canvas_height = img_shape[0]  
            
            canvas = tk.Canvas(master, width=canvas_width*1.5, height=canvas_height*1.5)

            # canvas = tk.Canvas(master, width=400*3, height=300*3)
            #canvas = tk.Canvas(master)
            canvas.grid(row=row, column=col, columnspan=2, padx=5, pady=5)  # Adjusted padding and grid position
            self.canvases.append(canvas)
  

        # Create a single scale widget for all images
        self.scale = tk.Scale(master, orient=tk.HORIZONTAL, from_=0, to=self.img_data[0].shape[2] - 1,
                              command=self.scale_slider)
        self.scale.grid(row=6, column=0, columnspan=1, padx=5, pady=10)  # Adjusted padding and grid position

        # Create dropdown menu for selecting different lists of images
        self.dropdown_var = tk.StringVar(master)
        self.dropdown_var.set('List 1')  # Default dropdown option
        self.dropdown_menu = tk.OptionMenu(master, self.dropdown_var, *['List {}'.format(i + 1) for i in range(len(nifti_files_lists))], command=self.change_list)
        self.dropdown_menu.grid(row=6, column=1, columnspan=1, padx=5, pady=5)  # Adjusted padding and grid position

        # Create dropdown menu for selecting the fourth image
        self.fourth_img_var = tk.StringVar(master)
        self.fourth_img_var.set(self.selected_nifti_files[3])  # Default fourth image
        self.fourth_img_menu = tk.OptionMenu(master, self.fourth_img_var, *self.selected_nifti_files[3:], command=self.change_fourth_image)
        self.fourth_img_menu.grid(row=3, column=3, columnspan=1, padx=5, pady=5)  # Adjusted padding and grid position
        self.fourth_img_menu.config(font=("Arial", 16, 'bold'))

        # Create dropdown menu for selecting view (axial or sagittal)
        self.view_var = tk.StringVar(master)
        self.view_var.set('Axial')  # Default view
        self.view_menu = tk.OptionMenu(master, self.view_var, 'Axial', 'Sagittal', command=self.change_view)
        self.view_menu.grid(row=3, column=4, columnspan=1, padx=5, pady=5)  # Adjusted padding and grid position

        # Create dropdown menus for selecting colormaps for each image
        self.colormap_vars = []
        self.colormap_menus = []
        for i in range(4):
            row = (i // 2) * 3 + 2  # Calculate row index
            col = (i % 2) * 3   # Calculate column index
            colormap_var = tk.StringVar(master)
            colormap_var.set('inferno')  # Default colormap
            colormap_menu = tk.OptionMenu(master, colormap_var, 'viridis', 'plasma', 'inferno', 'Greys_r', command=lambda value, index=i: self.change_colormap(value, index))
            colormap_menu.grid(row=row, column=col, columnspan=1, padx=5, pady=5)  # Adjusted padding and grid position
            self.colormap_vars.append(colormap_var)
            self.colormap_menus.append(colormap_menu)

        # Display initial slices
        self.display_slices()

        # Create labels to display pixel values
        self.pixel_value_labels = []
        for i in range(4):
            row = (i // 2) * 3 + 2  # Calculate row index for labels
            col = (i % 2) * 3 + 1 # Calculate column index for labels
            label = tk.Label(master, text="", width=20)
            label.grid(row=row, column=col, padx=5, pady=5)  # Adjusted padding and grid position
            self.pixel_value_labels.append(label)

        # Bind mouse events to canvases
        for canvas in self.canvases:
            canvas.bind("<Motion>", self.show_pixel_value)
            canvas.bind("<Leave>", self.clear_pixel_value)

        # Create text widget for displaying quality metrics
        self.text_widget = tk.Text(master, height=10, width=30, wrap=tk.WORD)
        self.text_widget.grid(row=0, column=7, rowspan=2, padx=5, pady=5)

        # Create text widget for explanations of the fourth image
        self.text_widget2 = tk.Text(master, height=15, width=30, wrap=tk.WORD)
        self.text_widget2.grid(row=4, column=5, columnspan=2, padx=5, pady=5)

        # Display initial slices and file contents
        self.display_slices()
        self.load_files_txt(files_txt[0])
        self.exp_text()

        # Titles for each image
        self.image_titles = ["Signal", "T1 image", "Standard deviation"]

        # Create labels for image titles
        self.image_title_labels = []
        for i in range(3):
            row = (i // 2) * 3  # Calculate row index for title labels
            col = (i % 2) * 3 # Calculate column index for title labels
            title_label = tk.Label(master, text=self.image_titles[i], font=("Arial", 16, 'bold'))
            title_label.grid(row=row, column=col, padx=10, pady=5)  # Adjusted padding and grid position
            self.image_title_labels.append(title_label)

        # Create a standalone canvas for histogram
        self.standalone_canvas = tk.Canvas(master, width=420, height=370)
        self.standalone_canvas.grid(row=1, column=5, columnspan=2, padx=5, pady=5)  # Adjusted padding and grid position

        # Initialize histogram plot
        px = 1/plt.rcParams['figure.dpi']  # pixel in inches
        self.figure = plt.figure(figsize=(400*px, 350*px))
        self.ax = self.figure.add_subplot(111)
        self.ax.set_xlabel('Values', fontsize=6)
        self.ax.set_ylabel('Frequency')
        self.ax.set_title('Histogram of Values')
        self.plot_img = None
        self.plot_hist = None

        # Display initial histogram
        self.update_histogram('1 CNR:')

        # Create dropdown menu for selecting different lists of values for histogram
        self.dropdown_var2 = tk.StringVar(master)
        self.dropdown_var2.set('1 CNR:')  # Default dropdown option
        self.dropdown_menu = tk.OptionMenu(master, self.dropdown_var2, *['{} {}'.format(i + 1,self.line_labels[i][0]) for i in range(len(self.line_arrays))], command=self.update_histogram)
        self.dropdown_menu.grid(row=2, column=5, padx=5, pady=5, sticky="W")

        self.copy_button = tk.Button(master, text="Add to reference set", command=self.copy_selected_files)
        self.copy_button.grid(row=4, column=7, padx=5, pady=5)

 
    def load_nifti(self):
        """
        Load NIfTI files into the application. Also checks which image type is selected to be 
        displayed in the fourth frame, making corresponding settings/calculations.

        Returns:
        None
        """
        for i in range(4):
            self.img_data[i] = nib.load(folder_path+'/'+self.selected_nifti_files[i]).get_fdata()
            # nifti = nib.load(folder_path+'/'+self.selected_nifti_files[i])
            # self.img_data[i] = nifti.get_fdata()
            if i == 3:
                self.snr = True if self.selected_nifti_files[i].endswith('_snr.nii') else False
                self.snrd = True if self.selected_nifti_files[i].endswith('_snrD.nii') else False
                self.field = True if self.selected_nifti_files[i].endswith('_field.nii') else False
                self.fieldz = True if self.selected_nifti_files[i].endswith('_fieldZmap.nii') else False
                self.snrz = True if self.selected_nifti_files[i].endswith('_snrZmap.nii') else False
                self.snrdz = True if self.selected_nifti_files[i].endswith('_snrDZmap.nii') else False
                if self.sag_slice:  # Apply transformation only to the fourth image
                        self.img_data[i] = np.transpose(self.img_data[i], axes=(2, 1, 0))
                        self.img_data[i] = np.flip(self.img_data[i], axis=0)
                        self.display_average()
                if (not self.fieldz and not self.snrz and not self.snrdz and not self.field):                                       
                    self.sum_canvas.create_text(50, 120, text="=", fill="black", font=('Helvetica 15 bold')) 
                    self.sum_canvas.create_line(0, 150, 100, 150, arrow=tk.LAST)                 
                    self.div_canvas.create_line(30, 0, 30, 100, arrow=tk.LAST)
                    self.div_canvas.create_text(15, 50, text="÷", fill="black", font=('Helvetica 15 bold'))
                else: 
                    self.sum_canvas.delete('all')
                    self.div_canvas.delete('all')

    def display_slices(self):
        """
        Display current slices of the loaded images on the canvases. Also 
        checks which image type is selected to be displayed in the fourth 
        frame, making corresponding settings/calculations.

        Returns:
        None
        """
        for i, canvas in enumerate(self.canvases):   
            if i == 3 and (self.snrz or self.snrdz or self.fieldz):
                slice_img_tmp = self.img_data[i][:, :, self.current_slices[i]]
                z_field_lim = max(abs(self.img_data[i].min()),abs(self.img_data[i].max()))
                cmap = plt.colormaps.get_cmap('RdBu')                    
               
                a = 0.5
                b = 2.0  # Exponent for more curvature
                c = 0.5
                X, Y = slice_img_tmp.shape
                slice_img_tmp2 = np.zeros((X,Y))
                for x in range(X):
                    for y in range(Y):
                        if slice_img_tmp[x,y] > 0:
                            normalized_array = slice_img_tmp[x,y] / 3.0
                            slice_img_tmp2[x,y] = a * np.power(normalized_array, b) + c
                        elif slice_img_tmp[x,y] < 0:
                            normalized_array = abs(slice_img_tmp[x,y]) / 3.0
                            slice_img_tmp2[x,y] = 1 - (a * np.power(normalized_array, b) + c )  
                        else:
                            slice_img_tmp2[x,y] = 0.5                             
                colored_img = cmap(slice_img_tmp2)

            else:
                slice_img_tmp = self.img_data[i][:, :, self.current_slices[i]]
                # Apply selected colormap
                cmap = self.colormaps[i]
                colored_img = cmap(slice_img_tmp / self.img_data[i].max())  # Apply colormap to image data

            # Convert colored image to RGB format
            colored_img_rgb = (colored_img[:, :, :3] * 255).astype('uint8')
            colored_img_pil = Image.fromarray(colored_img_rgb)

            # Resize the Image using resize method
            new_sizeratio = min(400 / slice_img_tmp.shape[1], 300 / slice_img_tmp.shape[0])
            resized_image = colored_img_pil.resize((int(slice_img_tmp.shape[1] * 1.5), int(slice_img_tmp.shape[0] * 1.5)), Image.NEAREST)
            colored_img_tk = ImageTk.PhotoImage(resized_image)
            
            # Delete previous image from canvas
            canvas.delete('all')

            # Display new image on canvas
            canvas.create_image(0, 0, anchor=tk.NW, image=colored_img_tk)
            canvas.image = colored_img_tk  # Keep a reference to prevent garbage collection

        # Check if there are values larger than 10 in any slice of the fourth image
        fourth_img_values = self.img_data[3]
        slices_with_large_values = []
        for slice_num in range(fourth_img_values.shape[2]):
            if np.any((fourth_img_values[:, :, slice_num] > 0) & (fourth_img_values[:, :, slice_num] < 1)):
                slices_with_large_values.append(slice_num)

        if slices_with_large_values and (not self.fieldz) and (not self.snrz) and (not self.snrdz) and (not self.field) and (not self.sag_slice):
            self.warning_label.config(text=f"Warning: snr < 1 found in slice(s): {', '.join(map(str, slices_with_large_values))}")
        else:
            self.warning_label.config(text="")  # Clear the warning message
            
    def scale_slider(self, value):
        """
        Change the slice displayed using the scale widget for all images.

        value: str, the new slice value

        Returns:
        None
        """
        self.current_slices = [int(value), int(value), int(value), int(value)]
        self.display_slices()

    def change_colormap(self, value, index):
        """
        Change the colormap for a specific image when the selection changes.

        value: str, the new colormap name
        index: int, index of the image to change colormap for

        Returns:
        None
        """
        self.colormaps[index] = plt.cm.get_cmap(value)
        self.display_slices()

    def change_list(self, value):
        """
        Update the selected NIfTI files based on the selected image in the dropdown option.

        value: str, the selected dropdown option

        Returns:
        None
        """
        self.list_index = int(value.split()[-1]) - 1
        self.default_nifti_files = self.nifti_files_lists[self.list_index]
        self.selected_nifti_files = self.default_nifti_files.copy()

        # Update options in the fourth dropdown menu
        self.fourth_img_var.set(self.default_nifti_files[3])  # Set fourth image to default value from the new list
        self.update_fourth_img_menu_options()

        self.load_nifti()
        self.display_slices()

        # Update command for fourth dropdown menu to call change_fourth_image with the updated value
        for index, img in enumerate(self.default_nifti_files[3:]):
            self.fourth_img_menu['menu'].entryconfig(index, command=lambda value=img: self.change_fourth_image(value))

        self.load_files_txt(files_txt[self.list_index])
        self.update_histogram(self.dropdown_var2.get())

    def update_fourth_img_menu_options(self):
        """
        Update the options in the fourth dropdown menu based on the selected image.

        Returns:
        None
        """
        self.fourth_img_menu['menu'].delete(0, 'end')
        for img in self.default_nifti_files[3:]:
            self.fourth_img_menu['menu'].add_command(label=img, command=tk._setit(self.fourth_img_var, img))

    def change_fourth_image(self, value):
        """
        Change the fourth image to the selected option.

        value: str, the name of the selected image option

        Returns:
        None
        """
        self.selected_nifti_files[3] = value
        self.load_nifti()
        self.display_slices()
        self.fourth_img_var.set(str(self.selected_nifti_files[3]))
        self.exp_text()

    def change_view(self, value):
        """
        Change the view (sagittal or axial) based on the selection.

        value: str, the selected view option

        Returns:
        None
        """
        self.sag_slice = True if value == 'Sagittal' else False
        self.load_nifti()
        self.display_slices()

    def show_pixel_value(self, event):
        """
        Show the pixel value of the image where the cursor is located on the canvas.
        Depending on the image in the fourth canvas, a certain value matrix is used.

        event: Tkinter event, the event is triggered by the cursor location

        Returns:
        None
        """
        canvas = event.widget
        index = self.canvases.index(canvas)
        slice_x = int(event.x / (canvas.winfo_width() / self.img_data[index].shape[1]))
        slice_y = int(event.y / (canvas.winfo_height() / self.img_data[index].shape[0]))        
        value = self.img_data[index][slice_y, slice_x, self.current_slices[index]]
        self.pixel_value_labels[index].config(text=f"Pixel Value: {value}")

    def clear_pixel_value(self, event):
        """
        Clear the displayed pixel value on the canvas.

        event: Tkinter event, the event is triggered by the cursor location

        Returns:
        None
        """
        canvas = event.widget
        index = self.canvases.index(canvas)
        self.pixel_value_labels[index].config(text="") 

    def analyse_files_txt(self, filename):
        """
        Analyse txt files and extract data to append to lists and
        gather data from all images in the selected dataset.

        filename: str, name of the txt file to analyse

        Returns:
        None
        """
        with open(filename, 'r') as file:
            lines = file.readlines()
            for i, line in enumerate(lines):
                entries = line.strip().split()
                if len(entries) >= 2:
                    value = float(entries[1])
                    if i not in self.line_arrays:
                        self.line_arrays[i] = []
                    self.line_arrays[i].append(value)

                    value = str(entries[0])
                    if i not in self.line_labels:
                        self.line_labels[i] = []
                    self.line_labels[i].append(value)

    def load_files_txt(self, filename):
        """
        Load contents of a txt file into the text widget in a certain format.

        filename: str, name of the txt file to load

        Returns:
        None
        """
        self.text_widget.delete('1.0', tk.END)            
        with open(filename, 'r') as file:    
            lines = file.readlines()
            self.curr_line_arrays = []
            self.curr_line_labels = []
            for i, line in enumerate(lines):
                entries = line.strip().split()
                if len(entries) >= 2:
                    value = float(entries[1])
                    self.curr_line_arrays.append(value)
                    value = str(entries[0])
                    self.curr_line_labels.append(value)

        file_contents = ""
        for idx in range(len(self.curr_line_labels)):
            file_contents = file_contents + str(self.curr_line_labels[idx]) + ' ' + "{:.4}".format(self.curr_line_arrays[idx]) + '\n'
    
        self.text_widget.insert(tk.END, file_contents)

    def update_histogram(self, selected_value):
        """
        Update the histogram based on the selected quality metric value from the 
        dropdown menu.

        selected_value: str, the selected dropdown value

        Returns:
        None
        """
        # Clear previous histogram
        self.ax.clear()
        if self.plot_hist:
            self.plot_hist.get_tk_widget().destroy()

        # Get the selected list of values from line_arrays
        selected_list_index_line = int(selected_value.split()[0]) - 1
        selected_values = self.line_arrays[selected_list_index_line]
        
        # Plot the histogram
        bins = np.linspace(min(selected_values)-(max(selected_values)-min(selected_values))/10, max(selected_values)+(max(selected_values)-min(selected_values))/10, 50)
        bar_value_to_label = self.curr_line_arrays[selected_list_index_line]
        patch_index = np.digitize([bar_value_to_label], bins)[0]
        s = pd.Series(selected_values)
        self.plot_hist = s.plot(kind='hist', bins=bins, color='orange')
        self.plot_hist.patches[patch_index-1].set_color('b')

        # Get the maximum y-value of the histogram
        max_y_value = 0
        for idx in range(len(self.plot_hist.patches)):
            if self.plot_hist.patches[idx-1].get_height() > max_y_value:
                max_y_value = self.plot_hist.patches[idx-1].get_height()

        # Add the arrow
        x_bin = self.plot_hist.patches[patch_index-1].get_x() + 0.5*self.plot_hist.patches[patch_index-1].get_width()
        y_bin = self.plot_hist.patches[patch_index-1].get_height()
        self.ax.annotate("", xy=(x_bin, y_bin + 1), xycoords='data',xytext=(x_bin, y_bin + 3), textcoords='data',arrowprops=dict(arrowstyle="->",connectionstyle="arc3"),)

        # Set labels and title
        self.ax.set_xlabel(str(self.line_labels[selected_list_index_line][0]))
        self.ax.set_ylabel('Frequency')
        self.ax.set_title('Histogram of ' + str(self.line_labels[selected_list_index_line][0]))
        self.ax.set_ylim(0,max_y_value + 5)

        # Convert the matplotlib plot to a tkinter-compatible image
        self.figure.canvas.draw()
        self.plot_hist = FigureCanvasTkAgg(self.figure, self.standalone_canvas)
        self.plot_hist.draw()
        self.plot_hist.get_tk_widget().pack()

    def display_average(self):
        """
        Calculate the average of the fourth image's slices and display as color scale on the left 
        side of the image.

        Returns:
        None
        """
        fourth_img_values = self.img_data[3]
        averages_per_slice_tmp = []

        for slice_num in range(fourth_img_values.shape[0]):
            current_slice = fourth_img_values[slice_num, :, :]
            nonzero_values = current_slice[current_slice != 0]

            if len(nonzero_values) > 0:
                average_nonzero = np.mean(nonzero_values)
                averages_per_slice_tmp.append(average_nonzero)
            else:
                # Handle case where slice has no nonzero values
                averages_per_slice_tmp.append(0)  

        self.averages_per_slice = np.array(averages_per_slice_tmp)

        # Add the averages as colorscale on the left side of the image
        i = 0
        for avg in self.averages_per_slice:
            self.img_data[3][i,0:4,:] = avg
            i += 1

    def exp_text(self):
        """
        Load explanation into the text widget.

        Returns:
        None
        """
        self.text_widget2.delete('1.0', tk.END) 
        if self.snr:           
            text_contents = ("Image shows signal-to-noise ratio (snr) per brain region." 
            "From purple to pink to yellow to white means a larger snr," 
            "normalized to the maximum snr value present in the image.")
        elif self.snrd:
            text_contents = ("Image shows Dietrichs signal-to-noise ratio (snrD) per brain region." 
            "From purple to pink to yellow to white means a larger snrD," 
            "normalized to the maximum snr value present in the image.")
        elif self.field:
            text_contents = ("Image shows the absolute deformation field of the registration" 
            "from the image to the MNI brain. From purple to pink to yellow to white" 
            "means a larger absolute deformation, normalized to the maximum snr value present in the image.")
        elif self.fieldz:
            text_contents = ("Image shows the z-score of the absolute deformation field relative to all input images."
            "White is a z-score of 0, negative scores are exponentially getting darker red, so that -2 is medium" 
            "red and -3 is dark red, positive scores are exponentially getting darker blue, so that 2 is medium"
            " blue and 3 is dark blue.")
        elif self.snrz:
            text_contents =("Image shows the z-score of the signal-to-noise ratio (snr) per brain region relative to" 
            "all input images. White is a z-score of 0, negative scores are exponentially getting darker red, "
            "so that -2 is medium red and -3 is dark red, positive scores are exponentially getting darker blue,"
            " so that 2 is medium blue and 3 is dark blue.")
        elif self.snrdz:
            text_contents = ("Image shows the z-score of Dietrichs signal-to-noise ratio (snrD) per brain region relative" 
            "to all input images. White is a z-score of 0, negative scores are exponentially getting darker red," 
            "so that -2 is medium red and -3 is dark red, positive scores are exponentially getting darker blue," 
            "so that 2 is medium blue and 3 is dark blue.")
        else:
            text_contents = []
        self.text_widget2.insert(tk.END, text_contents)

    def copy_selected_files(self):
        """
        Copy the currently selected NIfTI files from the combined folder to the reference set folder.
        """
        for filename in self.nifti_files_lists[self.list_index]:
            shutil.copy(os.path.join(combined_folder, filename), os.path.join(reference_set, filename))
            identifier, tag = filename.split("_", 1)
        shutil.copy(os.path.join(combined_folder, identifier+'_total_fieldZmap.nii'), os.path.join(reference_set, identifier+'_total_fieldZmap.nii'))
        print(os.path.join(combined_folder, identifier+'_total_fieldZmap.nii'))
        print()
        print(f"Copied {self.nifti_files_lists[self.list_index]} to {reference_set}")
        self.copy_called = True


###########


if __name__ == "__main__":
    # Main loop to run the GUI
    root = tk.Tk()
    root.title("NIfTI Viewer")

    # Initialize GUI with all lists of file names
    viewer = NiftiViewer(root, nifti_data, files_txt, snrlist_txt, snrDlist_txt,files_txt_ref)  

    def on_closing():
        if viewer.copy_called:
            ##Calculate new zmap values
            print("Updating average and std of reference set for z-maps")
            remove_identical_struct_nifti(reference_set)
            files_txt_ref, snrlist_txt_ref, snrDlist_txt_ref = qm_files_reference(reference_set)
            avg_reg, std_reg, region_arrays, region_values = calculate_Zmap(snrlist_txt_ref)
            avg_reg2, std_reg2, region_arrays2, region_values2 = calculate_Zmap(snrDlist_txt_ref)
            save_z_maps(avg_reg, std_reg, region_values,'total_zmap_SNR.txt')
            save_z_maps(avg_reg2, std_reg2, region_values2,'total_zmap_SNRD.txt')
            calculate_z_def_fields('total_zmap_fields.txt')
        shutil.rmtree(combined_folder)
        root.quit()

    # Bind the window closing event to the on_closing function
    root.protocol("WM_DELETE_WINDOW", on_closing)

    # Window can also be closed using the escape button
    root.bind("<Escape>", lambda x: root.destroy())
    root.mainloop()