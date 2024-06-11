import json
import os
import shutil

# Function to convert BDD100K det20 label to YOLO format
def bdd100k_to_yolo(bdd_label_path, yolo_output_path, class_mapping):
    # Load BDD100K det20 JSON file
    with open(bdd_label_path, 'r') as f:
        bdd_data = json.load(f)

    # Iterate through each image in the BDD100K det20 data
    for image_data in bdd_data:
        # Get image file name and path
        name = image_data['name']
        print(name)
        if 'labels' not in image_data:
            #shutil.move(f'E:/bdd100k/images/100k/train/{name}', f'E:/bdd100k/images/100k/image_without_label/{name}')
            continue
        image_filename = os.path.basename(image_data['name'])

        # Open YOLO output file for writing
        yolo_output_file = open(os.path.join(yolo_output_path, os.path.splitext(image_filename)[0] + '.txt'), 'w')

        # Iterate through each object annotation in the image
        for obj in image_data['labels']:
            # Get class label and bounding box coordinates
            class_label = obj['category']
            bbox = obj['box2d']

            # Map class label to YOLO index
            if class_label in class_mapping:
                class_index = class_mapping[class_label]
            else:
                # Skip if class label not in mapping
                continue

            # Convert bounding box coordinates to YOLO format
            x_center = (bbox['x1'] + bbox['x2']) / (2.0 * 1280)
            y_center = (bbox['y1'] + bbox['y2']) / (2.0 * 720)
            width = (bbox['x2'] - bbox['x1']) / 1280
            height = (bbox['y2'] - bbox['y1']) / 720

            # Write YOLO formatted line to output file
            yolo_output_file.write(f"{class_index} {x_center} {y_center} {width} {height}\n")

        # Close YOLO output file
        yolo_output_file.close()

# Example usage:
# Define BDD100K det20 label file path
bdd_label_path = 'E:/bdd100k/labels/bdd100k_labels_images_val.json'

# Define output directory for YOLO format labels
yolo_output_path = 'E:/bdd100k/labels/val/'

# Define class mapping from BDD100K classes to YOLO indices
class_mapping = {
    'pedestrian': 0,
    'rider': 1,
    'car': 2,
    'truck': 3,
    'bus': 4,
    'train': 5,
    'motorcycle': 6,
    'bicycle': 7,
    'traffic light': 8,
    'traffic sign': 9
}

# Convert BDD100K det20 labels to YOLO format
bdd100k_to_yolo(bdd_label_path, yolo_output_path, class_mapping)