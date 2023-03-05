# Dependencies
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm  # low overhead progress bar
import os  # navigate directories
from PIL import Image  # read images
import tensorflow as tf  # tensorflow
import tensorflow.keras as keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, BatchNormalization, Dropout, MaxPool2D, Input, Softmax, Activation, Flatten
from keras.utils.np_utils import to_categorical  # convert categorical form
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # image augmentation


# Function to compile all train/test data with outputs into array
def getdata(folder):
    # Initialize empty arrays for input/output
    X = []  # X is Greyscale image array input
    y = []  # y is Charpy classification output

    # Loops through all files in subdirectories in selected folder
    for ms_type in os.listdir(folder):
        if not ms_type.startswith('.'):  # Avoid .DStore hidden file for images on Mac
            # Assign labels based on subdirectory folder (Previously sorted)
            if ms_type in ['Sample01']:
                output = 1
            elif ms_type in ['Sample02']:
                output = 2
            elif ms_type in ['Sample03']:
                output = 3
            elif ms_type in ['Sample04']:
                output = 4
            else:  # Edge case, should not occur - All images are in folders
                output = 0

            # Loops through all image file in subdirectory
            for image_filename in tqdm(os.listdir(os.path.join(folder, ms_type))):
                if not ms_type.startswith('.'):  # Avoid .DStore hidden file for images on Mac
                    if not image_filename.startswith('.'):
                        img = Image.open(folder + '/' + ms_type + '/' + image_filename)
                        img_arr = np.asarray(img)
                        img_bw = tf.image.rgb_to_grayscale(img_arr)

                        # Append values to X,y
                        X.append(img_bw)
                        y.append(output)
                        # print(image_filename, output) # For debugging with filenames and output check
    return np.asarray(X), np.asarray(y)


# Define filepaths and directories
main_dir = '/Users/tszdabee/Desktop/FYP_Code/dataset-master/'  # Contains all files used in project
train_path = os.path.join(main_dir, 'TRAIN')
test_path = os.path.join(main_dir, 'TEST')
val_path = os.path.join(main_dir, 'VAL')

# Extract training and test data with getdata function defined
X_train, y_train = getdata(train_path)
X_test, y_test = getdata(test_path)
X_val, y_val = getdata(val_path)

# Data augmentation settings as "datagen"
train_datagen = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=30,
    fill_mode='reflect',
    # rescale = 1/255 # REMOVED rescale, efficientNet has in-built normalization
    # validation_split=0.2, #set validation split
    # brightness_range=[0.4,1],
    # zoom_range=[5,0.5]
)
train_datagen.fit(X_train)

# Normalization of image data, resize from 0-255 to 0-1
# X_train = X_train / 255 #removed, since rescale from Data Augmentation implements this
# X_test = X_test / 255 # REMOVED rescale, efficientNet has in-built normalization

# Categorical to encode into hot vectors. Note arg 0 represents no images
y_train_binary = to_categorical(y_train)
y_test_binary = to_categorical(y_test)
y_val_binary = to_categorical(y_val)

print("Shape for X_train: ", np.shape(X_train))
print("Shape for y_train: ", np.shape(y_train_binary))

# Create dataframe from labels
df_train = pd.DataFrame({'train_label': y_train})
df_test = pd.DataFrame({'test_label': y_test})

# Visualizing the dataset
# Plot no. of Training and Testing data
plt.figure(figsize=(10, 10))  # Define plot
sns.set_theme()  # Set figure theme
sns.set_style('darkgrid')  # Set figure style

plt.subplot(2, 1, 1)  # 2 rows, 1 column, 1st graph
axl1 = sns.countplot(x='train_label', data=df_train, palette=sns.color_palette("pastel"))  # Plot with palette
axl1.set_title("Training Image Distribution (n = " + str(len(y_train)) + ")")
axl1.set_xlabel('Sample #')

plt.subplot(2, 1, 2)  # 2 rows, 1 column, 2nd graph
axl2 = sns.countplot(x='test_label', data=df_test, palette=sns.color_palette("pastel"))
axl2.set_title("Testing Image Distribution (n = " + str(len(y_test)) + ")")
axl2.set_xlabel('Sample #')

plt.show()

# # Model 1: Basic CNN
# # three convolutional layers, three max-pooling layers, and two fully connected dense layers. use of batch normalization and dropout layers helps prevent overfitting and improve the generalization performance of the model.
# model = Sequential()
# model.add(Conv2D(64, 3, padding="same", activation="relu", input_shape=X_train.shape[1:]))
# model.add(MaxPool2D())
#
# model.add(Conv2D(64, 3, padding="same", activation="relu"))
# model.add(MaxPool2D())
#
# model.add(Conv2D(128, 3, padding="same", activation="relu"))
# model.add(MaxPool2D())
#
# model.add(Flatten())
# model.add(Dense(256, activation="relu"))
# model.add(Dropout(0.5)) # Dropout to prevent overfitting
# model.add(BatchNormalization())
# model.add(Dense(5, activation="softmax")) # Five outputs (including unused base case of index 0 to catch unexpected samples)

# Model 2: EfficientNetB0
# convert grayscale to rgb since EfficientNet pretrained on rgb
X_train = np.asarray([(np.dstack([X_train[i], X_train[i], X_train[i]])) for i in range(len(X_train))])
X_val = np.asarray([(np.dstack([X_val[i], X_val[i], X_val[i]])) for i in range(len(X_val))])
X_test = np.asarray([(np.dstack([X_test[i], X_test[i], X_test[i]])) for i in range(len(X_test))])
# Use EfficientNetB4
tmp = tf.keras.applications.EfficientNetB4(include_top=False,  # Now acts as feature extraction
                                           # Include_top lets you select if you want the fully connected final dense layers at end of the model.
                                           # This is usually what you want if you want the model to actually perform classification.
                                           # With include_top=True you can specify the parameter classes (defaults to 1000 for ImageNet).
                                           # With include_top=False, the model can be used for feature extraction,
                                           # Note that input_shape and pooling parameters should only be specified when include_top is False.
                                           weights='imagenet',
                                           # input_tensor=new_input,
                                           pooling='max',
                                           classes=5,  # number of output classes
                                           classifier_activation='softmax',
                                           drop_connect_rate=0.2)  # dropout of 0.5 instead of 0.2 default, prevent overfit
# Freeze pretrained model layers
for layer in tmp.layers:
    layer.trainable = False
# Define model
model = Sequential()
model.add(tmp)  # adding EfficientNetB4
model.add(Flatten())  # flattening the output feature maps, only use GlobalAveragePooling2D if
model.add(BatchNormalization())
# model.add(Dense(16, activation='relu'))
model.add(Dropout(rate=0.45))  # prevent overfit with regularization dropout
model.add(Dense(5, activation='softmax'))  # softmax for hot encode. CHANGE LATER for single prediction output

model.summary()

# Compile model
opt = keras.optimizers.Adam(learning_rate=1e-1)
model.compile(loss='categorical_crossentropy',
              optimizer="adam",
              metrics=['accuracy'])

# Training model
# checkpoint to save best models
filepath = "/Users/tszdabee/Desktop/FYP_Code/Model/EffNetB4.1e-1.weights.best.hdf5"
checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True,
                                             mode='max')  # Save new model if improves validation acc
es = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=8)  # stop running after 6 epochs no improvement
callbacks_list = [checkpoint, es]
history = model.fit(train_datagen.flow(X_train, y_train_binary, batch_size=16),
                    # 16 image augmentations during each epoch of training
                    epochs=40,
                    batch_size=32,
                    validation_data=(X_val, y_val_binary),  # from TEST: 80% training, 10% validation, 10% testing
                    callbacks=callbacks_list,  # save callbacks
                    verbose=1)

# Summarize Train/Validation Accuracy and Loss over epochs.
plt.style.use('ggplot')
plt.figure(figsize=(12, 12))
plt.subplot(2, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Training and Validation Accuracy')
plt.ylabel('Accuracy', fontsize=12)
plt.xlabel('Epoch', fontsize=12)
plt.legend(['train accuracy', 'validation accuracy'], loc='lower right', prop={'size': 12})

plt.subplot(2, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Training and Validation Loss')
plt.ylabel('Loss', fontsize=12)
plt.xlabel('Epoch', fontsize=12)
plt.legend(['train loss', 'validation loss'], loc='best', prop={'size': 12})

# Basic predict + evaluation with unseen test data
# model = keras.models.load_model("/Users/tszdabee/Desktop/FYP_Code/Model/EffNetB4.1e-1.weights.best.hdf5")
test_loss, test_acc = model.evaluate(X_test, y_test_binary, verbose=1)
print("The accuracy of the model is: ", test_acc)
print("The loss of the model is: ", test_loss)

# Further detailed evaluation for analysis
y_pred_prob = model.predict(X_test)  # predicts output in one hot encoded form [0.9 0.1 0]
y_pred = np.argmax(y_pred_prob, axis=1)  # select argmax with highest probability to match y_test
from sklearn.metrics import classification_report  # Confusion matrix


print(classification_report(y_test, y_pred)) # y_test instead of one hot encoded with original labels.


# plot architecture (Not working very well at the moment, will conduct manually with model summary)
# import visualkeras
# visualkeras.layered_view(model, legend=True).show() # display using your system viewer
# visualkeras.layered_view(model, legend=True, to_file='output.png') # write to disk
# visualkeras.layered_view(model, legend=True, to_file='output.png').show() # write and show
