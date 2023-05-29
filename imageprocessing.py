# Dependencies
import numpy as np
import matplotlib.pyplot as plt

#image augmentation
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import tensorflow as tf

image_filename = 'dog.jpg'#'dataset-raw/Sample03/image0012.tif'
img = load_img(image_filename)

#---convert the image to 3D array---
image_data = img_to_array(img)

#---convert into a 4-D array of 1 element of 3D array representing
# the image---
images_data = np.expand_dims(image_data, axis=0)

#---convert the image into greyscale (one dimensional)---
images_data = tf.image.rgb_to_grayscale(images_data)

datagen = ImageDataGenerator(width_shift_range=0.2,
                             height_shift_range=0.2,
                             horizontal_flip=True,
                             vertical_flip=True,
                             rotation_range=60,
                             #rescale= 1/255,
                             fill_mode = 'reflect'
                             #brightness_range=[0.15,2.0],
                             #zoom_range=[5,0.5]
                             )
train_generator = datagen.flow(images_data, batch_size=1)

# initialize plot axes
rows = 2
columns = 3
fig, axes = plt.subplots(rows,columns)
for r in range(rows):
    for c in range(columns):
        image_batch = train_generator.next()
        image = image_batch[0].astype('uint8')
        axes[r, c].imshow(image, cmap = 'gray') #plot images, grayscale
        axes[r, c].axis('off')  # remove axis labels
fig.set_size_inches(15,10)
plt.tight_layout()
plt.show() # show images

# NOTE: This augmentation does not rescale image by 1/255, need to do for final