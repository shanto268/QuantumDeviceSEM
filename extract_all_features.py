from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import pytesseract
import os
import datetime

def extract_scale(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Crop the image to the region of interest
    cropped = image[:50, 1050:]

    # Convert the image to grayscale
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

    # Threshold the image to isolate the white lines and text
    _, thresholded = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # Use pytesseract to extract the text
    text = pytesseract.image_to_string(thresholded)

    # Extract the number and unit from the text
    number = ''.join(filter(str.isdigit, text))
    unit = ''.join(filter(str.isalpha, text))

    # Convert the number to an integer
    number = int(number)

    # Convert the number to nanometers if the unit is not 'nm'
    if unit.lower() != 'nm':
        number *= 1000

    return number


# Load the model
sam = sam_model_registry["vit_h"](checkpoint="weights/sam_vit_h_4b8939.pth")

# Create a predictor
predictor = SamPredictor(sam)

# Ask for the image path
image_path = input("Enter the path to the image: ")

# Load the image
image = Image.open(image_path)

# Convert the image to a numpy array
image = np.array(image)

# Crop the bottom 80 pixels
crop_pixels = 80
scale_image = image[-crop_pixels:]
image = image[:-crop_pixels]

# Save the scale image
Image.fromarray(scale_image).save('scale.jpg')

# Set the image
predictor.set_image(image)

# Generate masks for an entire image
mask_generator = SamAutomaticMaskGenerator(sam)
masks = mask_generator.generate(image)

# Sort masks by area
masks.sort(key=lambda x: x['area'], reverse=True)

# Define the color to ignore
ignore_color = np.array([255, 0, 0])  # Red color

# Define the scale
phys_scale = extract_scale('scale.jpg') #nm
pixel_scale = 370 #measured

# Create results directory if it doesn't exist
image_name = os.path.splitext(os.path.basename(image_path))[0]
os.makedirs(f'results/{image_name}', exist_ok=True)

# Save a copy of the input image
Image.fromarray(image).save(f'results/{image_name}/{image_name}_copy.png')

# Display the masks
for i, mask in enumerate(masks):
    # Apply the mask to the image
    masked_image = image * mask['segmentation'][:, :, None]

    # Check if the masked image contains the ignore color
    if np.any(np.all(masked_image == ignore_color, axis=2)):
        continue

    # Calculate and display the size of the bounding box in nanometers
    bbox_size_nm = (mask['bbox'][2] * phys_scale / pixel_scale, mask['bbox'][3] * phys_scale / pixel_scale)
    
    fig, ax = plt.subplots(1)
    ax.imshow(masked_image)

    # Create a Rectangle patch for the bounding box
    bbox = patches.Rectangle((mask['bbox'][0], mask['bbox'][1]), mask['bbox'][2], mask['bbox'][3], linewidth=1, edgecolor='r', facecolor='none')

    # Add the patch to the Axes
    ax.add_patch(bbox)

    # Add the dimensions of the bounding box to the image
    ax.text(mask['bbox'][0], mask['bbox'][1] - 10, f'{bbox_size_nm[0]:.2f}nm', color='r')
    ax.text(mask['bbox'][0] + mask['bbox'][2] + 10, mask['bbox'][1] + mask['bbox'][3] / 2, f'{bbox_size_nm[1]:.2f}nm', color='r')

    # Remove the axes
    plt.axis('off')

    # Save the image
    plt.savefig(f'results/{image_name}/mask{i+1}_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}.png', bbox_inches='tight', pad_inches=0)

    plt.close()

