
import os
import random
import shutil

# Define the main directory
main_dir = '/Users/tszdabee/Desktop/FYP_Code/dataset-raw'
output_dir = '/Users/tszdabee/Desktop/FYP_Code/dataset-raw2'

# Define the names of the train, test, and validation directories
train_dir = 'TRAIN'
test_dir = 'TEST'
val_dir = 'VAL'

# Define the proportion of data to use for training, testing, and validation
train_proportion = 44/56
test_proportion = 6/56
val_proportion = 6/56

# Create the train, test, and validation directories
os.makedirs(os.path.join(output_dir, train_dir))
os.makedirs(os.path.join(output_dir, test_dir))
os.makedirs(os.path.join(output_dir, val_dir))

# Loop through each sample directory
for sample_dir in os.listdir(os.path.join(main_dir)):# s01,2,3,4
    if os.path.isdir(os.path.join(main_dir, sample_dir)): #open s01
        # Create a subdirectory for the sample in each of the train, test, and validation directories
        os.makedirs(os.path.join(output_dir, train_dir, sample_dir)) #main/train
        os.makedirs(os.path.join(output_dir, test_dir, sample_dir))
        os.makedirs(os.path.join(output_dir, val_dir, sample_dir))

        # Get a list of all the files in the sample directory
        file_list = os.listdir(os.path.join(main_dir, sample_dir))

        # Shuffle the file list
        random.shuffle(file_list)

        # Calculate the number of files to use for training, testing, and validation
        num_files = len(file_list)
        num_train = int(num_files * train_proportion)
        num_test = int(num_files * test_proportion)
        num_val = int(num_files * val_proportion)

        # Loop through the files in the shuffled list and copy them to the appropriate directory
        for i, file_name in enumerate(file_list):
            # Check file extension and ensure images selected
            if file_name.endswith(('.jpg', '.png', '.jpeg', '.tif')):
                if i < num_train:
                    shutil.copy(os.path.join(main_dir, sample_dir, file_name),
                                os.path.join(output_dir, train_dir, sample_dir, file_name))
                elif i < num_train + num_test:
                    shutil.copy(os.path.join(main_dir, sample_dir, file_name),
                                os.path.join(output_dir, test_dir, sample_dir, file_name))
                else:
                    shutil.copy(os.path.join(main_dir, sample_dir, file_name),
                                os.path.join(output_dir, val_dir, sample_dir, file_name))
