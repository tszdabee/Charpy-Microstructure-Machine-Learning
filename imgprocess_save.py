# Dependencies
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

#image augmentation
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import tensorflow as tf

folder_dir = "dataset-master/TRAIN/Sample03" # Select desired folder directory
filenames = Path(folder_dir).glob('*.tif') # Select all filenames with .tif extension

# Data augmentation settings as "datagen"
datagen = ImageDataGenerator(
     width_shift_range=0.1,
     height_shift_range=0.1,
     horizontal_flip=True,
     vertical_flip=True,
     rotation_range=70,
     fill_mode = 'reflect',
     #rescale = 1/255,
     #validation_split=0.2, #set validation split
     brightness_range=[0.4,1],
     #zoom_range=[5,0.5]
)

for image_filename in filenames: # Loop through all files in selected folder
    img = load_img(image_filename) # Create variable for selected image
    image_data = img_to_array(img) # Convert image to 3D array
    images_data = np.expand_dims(image_data, axis=0) # Reshape input into 4D array for 1 element, with 3D representing image
    image_bw = tf.image.rgb_to_grayscale(images_data)

    i = 0
    for batch in datagen.flow(image_bw, batch_size = 1, save_to_dir='dataset2-master/TRAIN/Sample03', save_prefix='aug_', save_format='jpeg'):
        i+=1
        if i>1: # Number of iterations
            break
print(image_bw)


"""
train_data_dir = "Sample01/aug" #dataset2-master/TRAIN/Sample01"
test_data_dir = "Sample01/aug
train_generator = datagen.flow_from_directory(
    train_data_dir,
    batch_size=1,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    test_data_dir,
    batch_size=1,
    class_mode='categorical',
    subset='training'
)"""
