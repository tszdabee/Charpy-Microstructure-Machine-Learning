import os
import numpy as np
import pandas as pd
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

# Set up the file paths
main_dir = '/Users/tszdabee/Desktop/FYP_Code/'
sample_dirs = [os.path.join(main_dir, 'dataset-raw', f'Sample{i:02}') for i in range(1, 13)]

# Load the dataset
df = pd.read_csv(os.path.join(main_dir, 'charpy_results.csv'))

# Preprocess the image data
X_img = []
temps = []
y=[]
for sample_dir in sample_dirs:
    sample_name = os.path.basename(sample_dir)
    sample_rows = df.loc[df['sample_name'] == sample_name]
    temps.append(sample_rows['temp_c'].values)
    y.append(sample_rows['impact_energy_j'].values)
    for i in range(56):
        img_path = os.path.join(sample_dir, f'image{i:04}.tif')
        sample_img = imread(img_path, as_gray=True)
        sample_img_resized = resize(sample_img, (256, 128)) # Resize to smaller dimensions
        sample_features = hog(sample_img_resized, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
        X_img.append(sample_features)
X_img = np.array(X_img)
temps = np.repeat(np.concatenate(temps), 56)
y =  np.repeat(np.concatenate(y), 56)

# Combine the image data and temperature data
X = np.hstack((X_img, temps.reshape(-1, 1)))

# Define the number of splits for cross-validation
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)


# Train and evaluate the random forest model using cross-validation
rf_mae_scores = []
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_y_pred = rf_model.predict(X_test)
    rf_mae = mean_absolute_error(y_test, rf_y_pred)
    rf_mae_scores.append(rf_mae)

print("Random Forest MAE Scores:", rf_mae_scores)
print("Random Forest Mean MAE:", np.mean(rf_mae_scores))

# Train and evaluate the SVR model using cross-validation
svr_mae_scores = []
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    svr_model = SVR(kernel='linear', C=100, gamma=0.1, epsilon=.1) #RBF leads to poor prediction of errors = 118 MAE
    svr_model.fit(X_train, y_train)
    svr_y_pred = svr_model.predict(X_test)
    svr_mae = mean_absolute_error(y_test, svr_y_pred)
    svr_mae_scores.append(svr_mae)

print("SVR MAE Scores:", svr_mae_scores)
print("SVR Mean MAE:", np.mean(svr_mae_scores))


# Visualize the predicted vs actual values
plt.scatter(y_test, rf_y_pred, alpha=0.5)
plt.plot(np.linspace(0, 400, 100), np.linspace(0, 400, 100), 'r--')
plt.xlabel('Actual Impact Energy (J)')
plt.ylabel('Predicted Impact Energy (J)')
plt.title('Random Forest Regression Model Performance (MAE=' + str(round(rf_mae, 4)) +')')
plt.show()

# Visualize the predicted vs actual values SVR
plt.scatter(y_test, svr_y_pred, alpha=0.5)
plt.plot(np.linspace(0, 400, 100), np.linspace(0, 400, 100), 'r--')
plt.xlabel('Actual Impact Energy (J)')
plt.ylabel('Predicted Impact Energy (J)')
plt.title('Support Vector Regression Model Performance (MAE=' + str(round(svr_mae, 4)) +')')
plt.show()

