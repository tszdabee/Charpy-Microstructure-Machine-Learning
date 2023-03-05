# Dependencies
import numpy as np
import matplotlib.pyplot as plt

#image augmentation
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import tensorflow as tf

image_filename = 'dataset-master/TRAIN/Sample03/image0011.tif'
img = load_img(image_filename)

#---convert the image to 3D array---
image_data = img_to_array(img)

#---convert into a 4-D array of 1 element of 3D array representing
# the image---
images_data = np.expand_dims(image_data, axis=0)

#---convert the image into greyscale (one dimensional)---
images_data = tf.image.rgb_to_grayscale(images_data)

""" #skimage preprocessing
from skimage import data, io, filters, exposure
p = io.imread("images/image.tif")

## 1. Image Sharpening
SharpImg = filters.unsharp_mask(p, radius = 20.0, amount = 1.0) # sharpening
SharpImg = exposure.adjust_gamma(SharpImg) # gamma correction on image

## Plotting Results
fig, ax = plt.subplots(nrows =1, ncols =2, sharex = True, figsize =(15,15))
ax[0].imshow(p, cmap = 'gray')
ax[0].set_title("Original", fontsize = 10)
ax[1].imshow(SharpImg, cmap ='gray')
ax[1].set_title("Sharpened",fontsize = 10)
plt.show()
"""
datagen = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                             horizontal_flip=True,
                             vertical_flip=True,
                             rotation_range=60,
                             #rescale= 1/255,
                             fill_mode = 'reflect'
                             #brightness_range=[0.15,2.0],
                             #zoom_range=[5,0.5]
                             )
train_generator = datagen.flow(images_data, batch_size=1)

#initialize plot axes
rows = 3
columns = 3
fig, axes = plt.subplots(rows,columns)
for r in range(rows):
    for c in range(columns):
        image_batch = train_generator.next()
        image = image_batch[0].astype('uint8')
        axes[r,c].imshow(image, cmap = 'gray') #plot images, grayscale
fig.set_size_inches(15,10)
plt.show() # show images

#NOTE: This augmentation does not rescale image by 1/255, need to do for final