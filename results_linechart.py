import matplotlib.pyplot as plt
import seaborn as sns

# Set the style to "darkgrid"
sns.set_style("darkgrid")

# Define the regressor models and corresponding MAE values
regressor = ['FCNN', 'RFR', 'SVR']
mae_hog = [11.8, 7.85, 9.173]
mae_vgg16 = [8.6, 5.1, 20.452]
mae_effnetb0 = [7.9078, 4.491, 23.4]

# Create the categorical line graph
plt.plot(regressor, mae_hog, marker='o', label='HOG features')
plt.plot(regressor, mae_vgg16, marker='o', label='VGG16 features')
plt.plot(regressor, mae_effnetb0, marker='o', label='EfficientNetB0 features')

# Add a red horizontal dotted line at 12.149 J
plt.axhline(y=12.149, color='red', linestyle='dotted', label='Baseline mean absolute error (12.1J)')

# Set the y-axis lower limit to 0
plt.ylim(bottom=0)

# Set the x-axis label and title
plt.xlabel('Regressor')
plt.ylabel('Mean Absolute Error (J)')
plt.title('Feature Extractor and Regressor performance comparison')

# Add a legend
plt.legend()

# Display the plot
plt.show()
