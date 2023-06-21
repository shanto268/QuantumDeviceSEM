import os
import datetime
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import pytesseract
from segment_anything import SamPredictor, sam_model_registry

def extract_scale(image_path):
    image = cv2.imread(image_path)
    cropped = image[:50, 1050:]
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    text = pytesseract.image_to_string(thresholded)
    number = ''.join(filter(str.isdigit, text))
    unit = ''.join(filter(str.isalpha, text))
    number = int(number)
    if unit.lower() != 'nm':
        number *= 1000
    return number

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

def load_model():
    sam = sam_model_registry["vit_h"](checkpoint="weights/sam_vit_h_4b8939.pth")
    return SamPredictor(sam)

def get_bbox(masked_image):
    mask = np.any(masked_image != 0, axis=-1)
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    bbox = (rmin, rmax, cmin, cmax)
    return bbox

def get_bbox_size(bbox, phys_scale, pixel_scale):
    bbox_height = bbox[1] - bbox[0]
    bbox_width = bbox[3] - bbox[2]
    bbox_size_nm = (bbox_width* phys_scale / pixel_scale, bbox_height * phys_scale / pixel_scale)
    return bbox_size_nm

def load_model():
    sam = sam_model_registry["vit_h"](checkpoint="weights/sam_vit_h_4b8939.pth")
    return SamPredictor(sam)

def print_message(message):
    print("\n" + "=" * 50)
    print(message)
    print("=" * 50 + "\n")

def process_image(image_path, pixel_scale=370, num_input=2):
    print_message("LOADING \'SEGMENT ANYTHING\' MODEL")
    predictor = load_model()
    image = Image.open(image_path)
    width, height = image.size
    scale_image = image.crop((0, height - 80, width, height))
    scale_image.save('scale.jpg')
    phys_scale = extract_scale('scale.jpg') #nm
    print_message("SCALE BAR LENGTH: {} nm".format(phys_scale))

    image = np.array(image.crop((0, 0, width, height - 80)))
    predictor.set_image(image)
    print_message("SELECT {} POINTS".format(num_input))
    plt.figure(figsize=(10,10))
    plt.imshow(image)
    input_points = np.array(plt.ginput(n=num_input, timeout=60, show_clicks=True))
    plt.axis('off')
    plt.show() 
    plt.close()

    input_label = np.ones(num_input, dtype=int)  # Create an array of ones with size num_input
    print("GENERATING MASKS TO EXTRACT FEATURE")
    masks, scores, logits = predictor.predict(
        point_coords=input_points,
        point_labels=input_label,
        multimask_output=False,
    )

    mask_input = logits[np.argmax(scores), :, :]  # Choose the model's best mask
    image_name = os.path.splitext(os.path.basename(image_path))[0]

    os.makedirs(f'results/{image_name}', exist_ok=True)
    Image.fromarray(image).save(f'results/{image_name}/{image_name}_copy.png')
    print_message("FEATURE EXTRACTION COMPLETE")
    original_shape = masks.shape
    new_shape = (original_shape[1], original_shape[2], 1)
    masks = masks.reshape(new_shape)
    masked_image = image * masks
    plt.imshow(masked_image)
    plt.axis('off')
    plt.show()

    bbox = get_bbox(masked_image)
    print_message("GENERATING BOUNDING BO")
    bbox_width, bbox_height = get_bbox_size(bbox, phys_scale, pixel_scale)
    fig, ax = plt.subplots(1)
    ax.imshow(masked_image)
    bbox_patches = patches.Rectangle((bbox[2], bbox[0]), bbox_width, bbox_height, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(bbox_patches)
    ax.text(bbox[2], bbox[0] - 10, f'{bbox_width:.2f}nm', color='r')
    ax.text(bbox[2] + bbox_width + 10, bbox[0] + bbox_height / 2, f'{bbox_height:.2f}nm', color='r')
    plt.axis('off')
    plt.savefig(f'results/{image_name}/mask_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}.png', bbox_inches='tight', pad_inches=0)
    print_message("SAVING MASKED IMAGE")
    plt.show()

if __name__ == "__main__":
    print_message("QUANTUM DEVICE SEM IMAGE ANALYSIS PROGRAM")
    image_path = input("Enter the path to the image: ")
    pixel_scale = input("Enter the pixel scale (default is 370): ") or 370
    num_input = input("Enter the number of inputs: ")
    process_image(image_path, int(pixel_scale), int(num_input))
