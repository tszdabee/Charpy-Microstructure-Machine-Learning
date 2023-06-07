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
from keras.layers import Dense, Conv2D, BatchNormalization, Dropout, Input, Flatten, Concatenate
from tensorflow.keras.losses import MeanAbsoluteError #losses
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # image augmentation
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.utils.class_weight import compute_class_weight
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.applications.vgg16 import VGG16 #VGG16 instead of EffNetB0

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

# Denormalize train/test labels for data visualization
df_train['train_label_plot'] = df_train['train_label'].multiply(scaler.data_range_[0]).round(0)
df_test['test_label_plot'] = df_test['test_label'].multiply(scaler.data_range_[0]).round(0)


# Visualizing the dataset
# Plot no. of Training and Testing data
plt.figure(figsize=(15, 5))  # Define plot
sns.set_theme()  # Set figure theme
sns.set_style('darkgrid')  # Set figure style

plt.subplot(1, 2, 1)  # 1 rows, 2 column, 1st graph
axl1 = sns.countplot(x='train_label_plot', data=df_train, palette=sns.color_palette("pastel"))  # Plot with palette
axl1.set_title("Training Image Distribution (n = " + str(len(y_train)) + ")")
axl1.set_xlabel('Sample Charpy impact energy (J)')

plt.subplot(1, 2, 2)  # 1 rows, 2 column, 2nd graph
axl2 = sns.countplot(x='test_label_plot', data=df_test, palette=sns.color_palette("pastel"))
axl2.set_title("Testing Image Distribution (n = " + str(len(y_test)) + ")")
axl2.set_xlabel('Sample Charpy impact energy (J)')

plt.tight_layout()  # Adjust spacing between subplots
plt.show()

# Model 2: Transfer Learning EfficientNet/VGG16
# convert grayscale to rgb since EfficientNet pretrained on rgb
X_train = np.asarray([(np.dstack([X_train[i], X_train[i], X_train[i]])) for i in range(len(X_train))])
X_val = np.asarray([(np.dstack([X_val[i], X_val[i], X_val[i]])) for i in range(len(X_val))])
X_test = np.asarray([(np.dstack([X_test[i], X_test[i], X_test[i]])) for i in range(len(X_test))])
# input params shape definition
img_size = len(X_train[0])
img_input = Input(shape=(img_size, img_size, 3)) #image input shape
temp_input = Input(shape=(1,), name='temp_input') #define temp input shape
# # Use EfficientNetb0
base_model = tf.keras.applications.EfficientNetB0(include_top=False,  # Now acts as feature extraction
                                           weights='imagenet',
                                           # input_tensor=new_input,
                                           pooling='max',
                                           drop_connect_rate=0.2)  # dropout of 0.5 instead of 0.2 default, prevent overfit
# Use VGG16
# base_model = VGG16(include_top=False,  # Now acts as feature extraction
#                    weights='imagenet',
#                    input_tensor=img_input,
#                    pooling='max')

# Freeze pretrained model layers
for layer in base_model.layers:
    layer.trainable = False
# Define model (image branch)
features = base_model(img_input) # image branch
features = BatchNormalization()(features)
features = Dropout(rate=0.45)(features)
flat_features = Flatten()(features)

# Temperature input branch
temp_input = Input(shape=(1,))
normalized_temp = BatchNormalization()(temp_input)
temp_dense = Dense(516, activation='relu')(normalized_temp) # introduce dense layer before concat for higher weight
flat_features = Concatenate()([flat_features, temp_dense]) #concatenate temp feature at end of EffNet feature extraction

# Add fully connected layers
x = Dense(64, activation='relu')(flat_features)
x = Dense(32, activation='relu')(x)

output = Dense(1)(x)

model = tf.keras.Model(inputs=[img_input, temp_input], outputs=output) # 1st model for NN prediction
model2 = tf.keras.Model(inputs=[img_input, temp_input], outputs=flat_features) # 2nd model for SVR, RF prediction

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
es = keras.callbacks.EarlyStopping(monitor="val_mean_absolute_error", patience=20, restore_best_weights=True)  # stop running after 5 epochs no improvement
callbacks_list = [checkpoint, es]
train_generator = train_datagen.flow((X_train, X_train_temp), y_train,batch_size=8) # define generator for data augmentation on images, but not on temp
history = model.fit(train_generator,
                    epochs=100,
                    batch_size=32,
                    validation_data=([X_val, X_val_temp], y_val),  # from TEST: 80% training, 10% validation, 10% testing
                    callbacks=callbacks_list,  # save callbacks
                    verbose=1,
                    class_weight=class_weights_dict
                    )

def plot_hist(history):
    # train/validation error and loss
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    train_mae = [x*scaler.data_range_[0] for x in history.history['mean_absolute_error']] #plot mae
    val_mae = [x*scaler.data_range_[0] for x in history.history['val_mean_absolute_error']]
    axs[0].axhline(y=mae_baseline, color="lightcoral", linestyle="dashed")
    axs[0].plot(train_mae)
    axs[0].plot(val_mae)
    axs[0].set_title('Model Mean Absolute Error (' + str(round(np.min(val_mae), 4)) + 'J)')
    axs[0].set_ylabel('Mean Absolute Error (J) for Training and Validation')
    axs[0].set_xlabel('Epoch')
    axs[0].legend(['mae baseline (' + str(np.round(mae_baseline,3)) + 'J)', 'training', 'validation'], loc='upper right')
    train_loss = [x*scaler.data_range_[0] for x in history.history['loss']] #plot loss
    val_loss = [x*scaler.data_range_[0] for x in history.history['val_loss']]
    axs[1].axhline(y=mae_baseline, color="lightcoral", linestyle="dashed")
    axs[1].plot(train_loss)
    axs[1].plot(val_loss)
    axs[1].set_title('Model Loss for Training and Validation (' + str(round(np.min(val_loss), 4)) + 'J)')
    axs[1].set_ylabel('Loss (J)')
    axs[1].set_xlabel('Epoch')
    axs[1].legend(['mae baseline (' + str(np.round(mae_baseline,3)) + 'J)', 'training', 'validation'], loc='upper right')
    plt.show()
plot_hist(history)


# basic predict + evaluation with unseen test data
model = keras.models.load_model("/Users/tszdabee/Desktop/FYP_Code/Model/test.b0.hdf5", custom_objects={'MeanAbsoluteError': MeanAbsoluteError()})

test_loss, test_mae = model.evaluate([X_test, X_test_temp], y_test, verbose=0) #evaluate model
print("The loss of the model on unseen data is: ", test_loss*scaler.data_range_[0]) #inverse transform back to actual
print('The mean absolute error of the model on unseen data is:', test_mae*scaler.data_range_[0])

# trained model to predict test data (unseen FCNN)
y_pred = model.predict([X_test, X_test_temp])
y_test_norm = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(-1,) #inverse transform test and prediction values for visualization
y_pred_norm = scaler.inverse_transform(y_pred.reshape(-1, 1)).reshape(-1,)


#### SVR and RANDOM FOREST PREDICTION with Transfer Learning
# Get the flat features from model2
train_flat_features = model2.predict([X_train, X_train_temp])
val_flat_features = model2.predict([X_val, X_val_temp])
test_flat_features = model2.predict([X_test, X_test_temp])
# Train and evaluate SVR model
svr_model = SVR()
svr_model.fit(np.concatenate([train_flat_features, val_flat_features]), np.concatenate([y_train, y_val]))
svr_predictions = svr_model.predict(test_flat_features)
svr_mae = mean_absolute_error(y_test, svr_predictions)
# Train and evaluate RF model
rf_model = RandomForestRegressor()
rf_model.fit(np.concatenate([train_flat_features, val_flat_features]), np.concatenate([y_train, y_val]))
rf_predictions = rf_model.predict(test_flat_features)
rf_mae = mean_absolute_error(y_test, rf_predictions)
print("SVR MAE:", svr_mae*scaler.data_range_[0])
print("RF MAE:", rf_mae*scaler.data_range_[0])



# Create a figure with two subplots
fig, (ax, ax1, ax2) = plt.subplots(1, 3, figsize=(15, 5))

# Scatter plot for FCNN
ax.scatter(y_test_norm, y_pred_norm, s=7, alpha=0.3, label='Data Points') #create scatterplot
ax.plot(np.linspace(0, 400, 100), np.linspace(0, 400, 100), 'r--', label='Perfect Fit (1:1)')
ax.set_xlabel('Actual Impact Energy (J)')
ax.set_ylabel('Predicted Impact Energy (J)')
ax.set_title('EfficientNetB0 with FCNN Performance (MAE=' + str(round(test_mae*scaler.data_range_[0], 1)) + 'J)')
ax.legend()


# Scatter plot for Random Forest Regression Model
rf_scatter = ax1.scatter(y_test*scaler.data_range_[0], rf_predictions*scaler.data_range_[0], alpha=0.3, s=7, label='Data Points')
ax1.plot(np.linspace(0, 400, 100), np.linspace(0, 400, 100), 'r--', label='Perfect Fit (1:1)')
ax1.set_xlabel('Actual Impact Energy (J)')
ax1.set_ylabel('Predicted Impact Energy (J)')
ax1.set_title('EfficientNetB0 with RFR Performance (MAE=' + str(round(rf_mae*scaler.data_range_[0], 1)) +'J)')
ax1.legend()
ax1.set_ylim(ax.get_ylim())

# Scatter plot for Support Vector Regression Model
svr_scatter = ax2.scatter(y_test*scaler.data_range_[0], svr_predictions*scaler.data_range_[0], alpha=0.3, s=7, label='Data Points')
ax2.plot(np.linspace(0, 400, 100), np.linspace(0, 400, 100), 'r--', label='Perfect Fit (1:1)')
ax2.set_xlabel('Actual Impact Energy (J)')
ax2.set_ylabel('Predicted Impact Energy (J)')
ax2.set_title('EfficientNetB0 with SVR Performance (MAE=' + str(round(svr_mae*scaler.data_range_[0], 1)) +'J)')
ax2.legend()
ax2.set_ylim(ax.get_ylim())

# Display the plot
plt.tight_layout()
plt.show()











# UNFREEZE LAYERS TO FINE TUNE
# def unfreeze_model(model):
#     # We unfreeze the top 20 layers while leaving BatchNorm layers frozen
#     for layer in model.layers[-20:]:
#         if not isinstance(layer, BatchNormalization):
#             layer.trainable = True
#     optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
#     model.compile(loss='mean_absolute_error',
#               optimizer="adam",
#               metrics=[MeanAbsoluteError()])
#
# unfreeze_model(model)
# checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor="val_mean_absolute_error", verbose=1, save_best_only=True,
#                                              mode='min')  # Save new model if error decreases
# es = keras.callbacks.EarlyStopping(monitor="val_mean_absolute_error", patience=10, restore_best_weights=True)  # stop running after 5 epochs no improvement
# callbacks_list = [checkpoint, es]
# hist = model.fit(train_generator,
#                     epochs=30,
#                     batch_size=32,
#                     validation_data=([X_val, X_val_temp], y_val),  # from TEST: 80% training, 10% validation, 10% testing
#                     callbacks=callbacks_list,  # save callbacks
#                     verbose=1,
#                     class_weight=class_weights_dict
#                     )
# plot_hist(hist)