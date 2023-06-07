import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, filters, feature

def visualize_hog(image_path):
    # Step 1: Import the image
    image = io.imread(image_path)

    # Plot the original image
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis('off')

    # Step 2: Image gradient computation
    gray = color.rgb2gray(image)
    gradient_x = filters.sobel_h(gray)
    gradient_y = filters.sobel_v(gray)
    gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)

    # Plot the gradient magnitude
    plt.subplot(2, 2, 2)
    plt.imshow(gradient_magnitude, cmap='gray')
    plt.title("Gradient Magnitude")
    plt.axis('off')

    # Step 3: Image cell division
    cell_size = (8, 8)
    num_cells_x = gray.shape[1] // cell_size[1]
    num_cells_y = gray.shape[0] // cell_size[0]

    # Plot the divided cells
    plt.subplot(2, 2, 3)
    plt.imshow(gradient_magnitude, cmap='gray')
    for i in range(1, num_cells_x):
        plt.axvline(i * cell_size[1], color='lime', linewidth=0.3)
    for i in range(1, num_cells_y):
        plt.axhline(i * cell_size[0], color='lime', linewidth=0.3)
    plt.title("Cell Division")
    plt.axis('off')

    # Step 4: Gradient histogram computation
    _, hog_image = feature.hog(gray, visualize=True, cells_per_block=(1, 1))

    # Plot the HOG features
    plt.subplot(2, 2, 4)
    plt.imshow(hog_image, cmap='gray')
    plt.title("HOG Features")
    plt.axis('off')

    plt.tight_layout()
    plt.show()


visualize_hog('dog2.jpg')