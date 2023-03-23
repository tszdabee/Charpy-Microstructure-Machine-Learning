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
from keras.layers import Dense, Conv2D, BatchNormalization, Dropout, MaxPool2D, Input, Softmax, Activation, Flatten, Concatenate, Lambda
from keras.utils.np_utils import to_categorical  # convert categorical form
from tensorflow.keras.losses import MeanAbsoluteError, MeanAbsolutePercentageError #losses
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # image augmentation
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.utils.class_weight import compute_class_weight

# Function to compile all train/test data with outputs into array
def getdata(folder):
    # Initialize empty arrays for input/output
    X = []  # X is Greyscale image array input
    X_temp = [] # X_temp is corresponding temperature input
    y = []  # y is Charpy classification output

    # load charpy results into dataframe to compile
    df = pd.read_csv('charpy_results.csv')
    energy_dict = dict(zip(df['sample_name'], df['impact_energy_j'])) #dictionary with sample name as keys, impact energy as values.
    temp_dict = dict(zip(df['sample_name'], df['temp_c']))  # dictionary with sample name as keys, temp as values.

    # Loops through all files in subdirectories in selected folder
    for ms_type in os.listdir(folder):
        if not ms_type.startswith('.'):  # Avoid .DStore hidden file for images on Mac
            # Assign labels based on subdirectory folder (Previously sorted)
            if ms_type in energy_dict:
                output = energy_dict[ms_type]
                temp = temp_dict[ms_type]
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
                        X_temp.append(temp)
                        y.append(output)
                        
                        #print(image_filename, output, temp) # For debugging with filenames and output check
    return np.asarray(X), np.asarray(X_temp), np.asarray(y)

# function to calculate baseline mae
def calculate_mae_baseline(csv_filename):
    df = pd.read_csv(csv_filename)
    X = df[['temp_c']]
    y = df['impact_energy_j']
    # linear regression on data
    lin_reg = LinearRegression()
    lin_reg.fit(X, y)
    df['predicted'] = lin_reg.predict(X)
    # abs diff + mae
    df['abs_diff'] = abs(df['predicted'] - df['impact_energy_j'])
    mae = mean_absolute_error(df['impact_energy_j'], df['predicted'])
    return mae

mae_baseline = calculate_mae_baseline('charpy_results.csv')

# Define filepaths and directories
main_dir = '/Users/tszdabee/Desktop/FYP_Code/dataset-master/'  # Contains all files used in project
train_path = os.path.join(main_dir, 'TRAIN')
test_path = os.path.join(main_dir, 'TEST')
val_path = os.path.join(main_dir, 'VAL')

# Extract training and test data with getdata function defined
X_train, X_train_temp, y_train = getdata(train_path)
X_test, X_test_temp, y_test = getdata(test_path)
X_val, X_val_temp, y_val = getdata(val_path)

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
train_datagen.fit(X_train) #augment training data

#scale fit outputs, inverse scale later to obtain real values. helps converge faster
scaler = MinMaxScaler()
y_train = scaler.fit_transform(y_train.reshape(-1, 1)).reshape(-1,)
y_val = scaler.transform(y_val.reshape(-1, 1)).reshape(-1,)
y_test = scaler.transform(y_test.reshape(-1, 1)).reshape(-1,)

print("Shape for X_train: ", np.shape(X_train))
print("Shape for y_train: ", np.shape(y_train))

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
# input params shape definition
img_size = len(X_train[0])
img_input = Input(shape=(img_size, img_size, 3)) #image input shape
temp_input = Input(shape=(1,), name='temp_input') #define temp input shape
# Use EfficientNetB0
base_model = tf.keras.applications.EfficientNetB0(include_top=False,  # Now acts as feature extraction
                                           weights='imagenet',
                                           # input_tensor=new_input,
                                           pooling='max',
                                           drop_connect_rate=0.2)  # dropout of 0.5 instead of 0.2 default, prevent overfit
# Freeze pretrained model layers
for layer in base_model.layers:
    layer.trainable = False
# Define model (image branch)
x = base_model(img_input) # image branch
x = Flatten()(x)
x = BatchNormalization()(x)
x = Dropout(rate=0.45)(x)
temp_input = Input(shape=(1,)) #temperature branch
t = Dense(16, activation='relu')(temp_input)
x = Concatenate()([x, t]) #concatenate temp feature at end of EffNet feature extraction

output = Dense(1)(x)

model = tf.keras.Model(inputs=[img_input, temp_input], outputs=output)

model.summary()

# Find class weights (to solve data imbalance by giving more weight to minority classes)
class_weights = compute_class_weight('balanced', np.unique(y_train), y_train)
class_weights_dict = dict(enumerate(class_weights))

# Compile model
opt = keras.optimizers.Adam(learning_rate=1e-3)
model.compile(loss='mean_absolute_error',
              optimizer="adam",
              metrics=[MeanAbsoluteError()])

# Training model
# checkpoint to save best models
filepath = "/Users/tszdabee/Desktop/FYP_Code/Model/test.b0.hdf5"
checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor="val_mean_absolute_error", verbose=1, save_best_only=True,
                                             mode='min')  # Save new model if error decreases
es = keras.callbacks.EarlyStopping(monitor="val_mean_absolute_error", patience=5, restore_best_weights=True)  # stop running after 5 epochs no improvement
callbacks_list = [checkpoint, es]
train_generator = train_datagen.flow((X_train, X_train_temp), y_train,batch_size=8) # define generator for data augmentation on images, but not on temp
history = model.fit(train_generator,
                    # 16 image augmentations during each epoch of training
                    epochs=50,
                    batch_size=32,
                    validation_data=([X_val, X_val_temp], y_val),  # from TEST: 80% training, 10% validation, 10% testing
                    callbacks=callbacks_list,  # save callbacks
                    verbose=1,
                    class_weight=class_weights_dict
                    )

# train/validation error and loss
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

train_mae = [x*scaler.data_range_[0] for x in history.history['mean_absolute_error']] #plot mae
val_mae = [x*scaler.data_range_[0] for x in history.history['val_mean_absolute_error']]
axs[0].axhline(y=mae_baseline, color="lightcoral", linestyle="dashed")
axs[0].plot(train_mae)
axs[0].plot(val_mae)
axs[0].set_title('Model Accuracy (val_mae=' + str(round(val_mae[-1], 4)) + 'J)')
axs[0].set_ylabel('Mean Absolute Error (J) for Training and Validation')
axs[0].set_xlabel('Epoch')
axs[0].legend(['mae baseline (' + str(mae_baseline) + ')', 'training', 'validation'], loc='upper left')
train_loss = [x*scaler.data_range_[0] for x in history.history['loss']] #plot loss
val_loss = [x*scaler.data_range_[0] for x in history.history['val_loss']]
axs[1].axhline(y=mae_baseline, color="lightcoral", linestyle="dashed")
axs[1].plot(train_loss)
axs[1].plot(val_loss)
axs[1].set_title('Model Loss for Training and Validation (' + str(round(val_loss[-1], 4)) + 'J)')
axs[1].set_ylabel('Loss (J)')
axs[1].set_xlabel('Epoch')
axs[1].legend(['mae baseline (' + str(mae_baseline) + ')', 'training', 'validation'], loc='upper right')
plt.show()

#original code
# loss = history.history['loss']
# epochs = range(1, len(loss)+1)
# plt.style.use('ggplot')
# plt.plot(epochs, loss, 'ro', label='Training loss')
# plt.legend()
# plt.show()


# basic predict + evaluation with unseen test data
model = keras.models.load_model("/Users/tszdabee/Desktop/FYP_Code/Model/test.b0.hdf5", custom_objects={'MeanAbsoluteError': MeanAbsoluteError()})
test_loss, test_mae = model.evaluate([X_test, X_test_temp], y_test, verbose=0) #evaluate model
print("The loss of the model on unseen data is: ", test_loss*scaler.data_range_[0]) #inverse transform back to actual
print('The mean absolute error of the model on unseen data is:', test_mae*scaler.data_range_[0])



# trained model to predict test data (unseen)
y_pred = model.predict([X_test, X_test_temp])
y_test_norm = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(-1,) #inverse transform test and prediction values for visualization
y_pred_norm = scaler.inverse_transform(y_pred.reshape(-1, 1)).reshape(-1,)
plt.scatter(y_test_norm, y_pred_norm) #create scatterplot
plt.plot([min(y_test_norm), max(y_test_norm)], [min(y_test_norm), max(y_test_norm)], 'r--')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.show()


# inverse scale back to nominal values (not really needed)
# y_train = scaler.inverse_transform(y_train.reshape(-1, 1)).reshape(-1,)
# y_val = scaler.inverse_transform(y_val.reshape(-1, 1)).reshape(-1,)
# y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(-1,)





# plot architecture (Not working very well at the moment, will conduct manually with model summary)
# import visualkeras
# visualkeras.layered_view(model, legend=True).show() # display using your system viewer
# visualkeras.layered_view(model, legend=True, to_file='output.png') # write to disk
# visualkeras.layered_view(model, legend=True, to_file='output.png').show() # write and show
