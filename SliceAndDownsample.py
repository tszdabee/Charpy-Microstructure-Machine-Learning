# This script conducts data preprocessing on raw data, slices and downsamples all images. Training data is randomly cropped
# Inputs:
# input_dir is input folder path
# output_dir is output folder path (Create blank folder beforehand)
# size is slicing pixels size x size
# downampling factor is how much to downsample images by X factor

# Dependencies
from PIL import Image # Python Imaging Library
import os
import random

# Call the function with the desired parameters
input_dir = '/Users/tszdabee/Desktop/FYP_Code/dataset-raw2'
output_dir = '/Users/tszdabee/Desktop/FYP_Code/dataset-master'
size = 570 # slice to 512x512 images - remainder will be discarded
factor = 1.5 # downsampling factor
num_crops = 5 # number of random crops
# 570/1.5 = output images of size 380x380

# Function to create number of random crops of images of desired size x size from source image, specify input and output folders
def randomCrops(input_folder, output_folder, size, num_crops):
    # Create output folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Obtain list of all images in input folder
    images = [f for f in os.listdir(input_folder) if f.endswith('.tif')]

    # Iterate images
    for image in images:
        img = Image.open(os.path.join(input_folder, image))

        # Convert images to RGB or L mode if it is in P mode
        if img.mode == 'P':
            img = img.convert('RGB')

        # Obtain image width, height
        width, height = img.size

        # Loop through specified number of random crops
        for i in range(num_crops):
            # Generate random top-left coordinate for crop, (0,width-size) but added 0.5*size to create boudning box for crops
            left = random.randint(size*0.5, width - 1.5*size)
            top = random.randint(size*0.5, height - 1.5*size)
            right = left + size
            bottom = top + size

            # Crop the slice from the original image
            slice = img.crop((left, top, right, bottom))

            # Save the slice to the output folder as JPEGs
            slice_name = image.split('.')[0] + '_crop' + str(i) + '.jpeg'
            slice.save(os.path.join(output_folder, slice_name), 'JPEG')

# Function to slice images into desired size x size, with inputs and output folders
def sliceImages(input_folder, output_folder, size):
    # Create output folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Obtain list of all images in input folder
    images = [f for f in os.listdir(input_folder) if f.endswith('.tif')]

    # Iterate images
    for image in images:
        img = Image.open(os.path.join(input_folder, image))

        # Convert images to RGB or L mode if it is in P mode
        if img.mode == 'P':
            img = img.convert('RGB')

        # Obtain image width, height
        width, height = img.size

        # Calculate number of slices in each direction, rounding down. Excess will be wasted data
        num_slices_x = int(width/size)
        num_slices_y = int(height/size)

        # Loop through all sliced images in x, y direction
        for i in range(num_slices_x):
            for j in range(num_slices_y):
                # Get bounds of the current slice
                left = i*size
                right = (i+1)*size
                top = j*size
                bottom = (j+1)*size

                # Crop the slice from the original image
                slice = img.crop((left, top, right, bottom))

                # Save the slice to the output folder as JPEGs
                slice_name = image.split('.')[0] + '_' + str(i) + '_' + str(j) + '.jpeg'
                slice.save(os.path.join(output_folder, slice_name), 'JPEG')

# Function to downsample all images in given folder path by the defined factor, while replacing existing images
def downsampleImages(folder_path, downsample_factor):
    # Loop through all files in folder_path
    for file_name in os.listdir(folder_path):
        # Check file extension and ensure images selected
        if file_name.endswith(('.jpg', '.png', '.jpeg', '.tif')):
            # Load image
            image = Image.open(os.path.join(folder_path, file_name))

            # Downsample the image by specified factor
            new_height, new_width = int(image.height // downsample_factor), int(image.width // downsample_factor)
            downsampled_image = image.resize((new_width, new_height))

            # Save downsampled image, replace original file
            downsampled_image.save(os.path.join(folder_path, file_name))

# Loop through each sample directory
for ttv_dir in os.listdir(os.path.join(input_dir)): #test, train validation folders
    # skip hidden file .DS_Store for mac
    if not ttv_dir.startswith('.'):
        # Create a subdirectory train, test, and validation directories
        os.makedirs(os.path.join(output_dir,ttv_dir))
        for sample_dir in os.listdir(os.path.join(input_dir, ttv_dir)):# s01,s02,s03,s04
            # skip hidden file .DS_Store for mac
            if not sample_dir.startswith('.'):
                # Create a subdirectory for the sample in each of the train, test, and validation directories
                os.makedirs(os.path.join(output_dir, ttv_dir, sample_dir))  # output/train_test_val/samples
                # Slice image to desired size
                sliceImages(input_folder=os.path.join(input_dir, ttv_dir, sample_dir), output_folder=os.path.join(output_dir, ttv_dir, sample_dir), size=size)
                # Apply randomCrops only to training images
                if ttv_dir == 'TRAIN':
                    randomCrops(input_folder=os.path.join(input_dir, ttv_dir, sample_dir), output_folder=os.path.join(output_dir, ttv_dir, sample_dir), size=size, num_crops=num_crops)
                # Downsample all images
                downsampleImages(folder_path=os.path.join(output_dir, ttv_dir, sample_dir), downsample_factor=factor)
