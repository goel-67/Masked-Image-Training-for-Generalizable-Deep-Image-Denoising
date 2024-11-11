import os
import cv2
import numpy as np
from skimage import util, io
from glob import glob

# Function to add a random combination of noise to the image
def add_random_combined_noise(image):
    # Apply a combination of Gaussian, Poisson, and speckle noise
    image = add_gaussian_noise(image)
    image = add_poisson_noise(image)
    image = add_speckle_noise(image)
    return image

# Function to add Gaussian noise
def add_gaussian_noise(image, mean=0, var=0.01):
    noisy_image = util.random_noise(image, mode='gaussian', mean=mean, var=var)
    return (noisy_image * 255).astype(np.uint8)

# Function to add Poisson noise
def add_poisson_noise(image):
    noisy_image = util.random_noise(image, mode='poisson')
    return (noisy_image * 255).astype(np.uint8)

# Function to add speckle noise
def add_speckle_noise(image):
    noisy_image = util.random_noise(image, mode='speckle')
    return (noisy_image * 255).astype(np.uint8)

# Create a folder to save noisy images
output_folder = 'noisy_images'
os.makedirs(output_folder, exist_ok=True)

# Path to the folder containing ground truth TIFF images
input_folder = 'testset/McM/HR'

tif_files = glob(os.path.join(input_folder, '*.tif'))

# Process each TIFF image
for file_path in tif_files:
    # Read the image
    image = io.imread(file_path)
    if image.ndim == 2:  # Convert grayscale to RGB
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # Apply a random combination of noise
    noisy_image = add_random_combined_noise(image)
    
    # Save the image with the same name in the output folder
    base_filename = os.path.splitext(os.path.basename(file_path))[0]
    output_path = os.path.join(output_folder, f'{base_filename}.png')
    cv2.imwrite(output_path, cv2.cvtColor(noisy_image, cv2.COLOR_RGB2BGR))
    print(f'Saved: {output_path}')

print('All noisy images have been created and saved.')
