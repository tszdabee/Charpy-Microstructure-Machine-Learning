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
from sklearn.model_selection import GridSearchCV # Hyperparameter tuning heatmap
import seaborn as sns
import glob

# Set up the file paths
main_dir = '/Users/tszdabee/Desktop/FYP_Code/'
sample_dirs = [os.path.join(main_dir, 'dataset-HOG', f'Sample{i:02}') for i in range(1, 13)]

# Load the dataset
df = pd.read_csv(os.path.join(main_dir, 'charpy_results.csv'))

# Preprocess the image data
X_img = []
temps = []
y=[]
for sample_dir in sample_dirs:
    sample_name = os.path.basename(sample_dir)
    sample_rows = df.loc[df['sample_name'] == sample_name]
    sample_temps = sample_rows['temp_c'].values
    sample_energies = sample_rows['impact_energy_j'].values
    for img_path in glob.glob(os.path.join(sample_dir, '*.jpeg')):
        sample_img = imread(img_path, as_gray=True)
        sample_img_resized = resize(sample_img, (256, 256))  # Resize to smaller dimensions
        sample_features = hog(sample_img_resized, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
        X_img.append(sample_features)
        temps.append(sample_temps)
        y.append(sample_energies)
X_img = np.array(X_img)
temps = np.concatenate(temps)
y = np.concatenate(y)

# Combine the image data and temperature data
X = np.hstack((X_img, temps.reshape(-1, 1)))

# Define the number of splits for cross-validation (Used for both tuning and final code)
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)


#svr random tuning
# Specify hyperparameter search space
param_grid = {'C': [0.01, 0.1, 1, 10, 100],
              'gamma': [1e-3, 0.01, 0.1, 1, 10],
              'epsilon': [0.01, 0.1, 1, 10]}

# Create SVR estimator
svr = SVR(kernel='linear')

# Perform grid search with cross-validation
svr_grid_search = GridSearchCV(svr, param_grid, scoring='neg_mean_absolute_error', cv=kf, verbose=3)
svr_grid_search.fit(X, y)
# The thing is that GridSearchCV, by convention, always tries to maximize its score so loss functions like MSE have to be negated.The unified scoring API always maximizes the score, so scores which need to be minimized are negated in order for the unified scoring API to work correctly. The score that is returned is therefore negated when it is a score that should be minimized and left positive if it is a score that should be maximized.
# Convention that higher values are better than lower.

# Print the best hyperparameters
print("Best SVR Hyperparameters:", svr_grid_search.best_params_)

# Extract results into a pandas DataFrame
results = pd.DataFrame(svr_grid_search.cv_results_)

# Create heatmap of mean test score by hyperparameters
scores_svr = svr_grid_search.cv_results_['mean_test_score'].reshape(len(param_grid['C']), len(param_grid['gamma']), len(param_grid['epsilon']))
scores_svr_mean = np.mean(scores_svr, axis=2) # Taking the mean across epsilon values for heatmap visualization
sns.heatmap(pd.DataFrame(scores_svr_mean, index=param_grid['C'], columns=param_grid['gamma']), annot=True, cmap='coolwarm', cbar_kws={'label': 'Negative Mean Absolute Test Score'})
plt.title("Support Vector Regression (SVR) Hyperparameter Tuning")
plt.ylabel('C')
plt.xlabel('Gamma')
plt.show()


#random forest hyperparam tuning
# Define hyperparameter ranges to search
param_grid = {
    'n_estimators': [32, 64, 128, 256, 516],
    'max_depth': [2, 4, 8, 16, 32]
}

# Create the random forest model
rf_model = RandomForestRegressor(random_state=42)

# Perform a grid search to find the best hyperparameters
rf_grid_search = GridSearchCV(rf_model, param_grid, cv=kf, scoring='neg_mean_absolute_error', verbose=3)
rf_grid_search.fit(X, y)

# Print the best hyperparameters
print("Best Random Forest Hyperparameters:", rf_grid_search.best_params_)

# Create a heatmap of the mean test scores for each combination of hyperparameters
scores = rf_grid_search.cv_results_['mean_test_score'].reshape(len(param_grid['n_estimators']), len(param_grid['max_depth']))
plt.imshow(scores, cmap='hot', interpolation='nearest')
plt.yticks(range(len(param_grid['max_depth'])), param_grid['max_depth'])
plt.xticks(range(len(param_grid['n_estimators'])), param_grid['n_estimators'])
plt.ylabel('Max Depth')
plt.xlabel('Number of Estimators')
plt.title("Random Forest Hyperparameter Tuning")
plt.colorbar().set_label('Negative Mean Absolute Test Score')
plt.show()


# USE OPTIMIZED HYPERPARAMETERS TO CONDUCT K FOLD CROSS VALIDATION
# Train and evaluate the random forest model using cross-validation
rf_mae_scores = []
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    rf_model = RandomForestRegressor(n_estimators=rf_grid_search.best_params_['n_estimators'], max_depth=rf_grid_search.best_params_['max_depth'], random_state=42)
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
    svr_model = SVR(kernel='linear', C=svr_grid_search.best_params_['C'], gamma=svr_grid_search.best_params_['gamma'], epsilon=svr_grid_search.best_params_['epsilon']) #RBF leads to poor prediction of errors = 118 MAE
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
plt.title('Random Forest Regression Model Performance (MAE=' + str(round(np.mean(rf_mae_scores), 4)) +')')
plt.show()

# Visualize the predicted vs actual values SVR
plt.scatter(y_test, svr_y_pred, alpha=0.5)
plt.plot(np.linspace(0, 400, 100), np.linspace(0, 400, 100), 'r--')
plt.xlabel('Actual Impact Energy (J)')
plt.ylabel('Predicted Impact Energy (J)')
plt.title('Support Vector Regression Model Performance (MAE=' + str(round(np.mean(svr_mae_scores), 4)) +')')
plt.show()