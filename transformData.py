import os
import json
import shutil

def transform_dataset(input_path, output_path):
    # Read metadata file
    metadata_path = os.path.join(input_path, 'meta', 'test.json')

    with open(metadata_path, 'r') as metadata_file:
        metadata = json.load(metadata_file)

    # Create output directory
    output_train_path = os.path.join(output_path, 'test')
    os.makedirs(output_train_path, exist_ok=True)

    # Move images to the new structure for training set
    for class_name, images in metadata.items():
        class_path = os.path.join(output_train_path, class_name)
        os.makedirs(class_path, exist_ok=True)

        for image_name in images:
            # Assuming image_name is in the format "churros/1061830.jpg"
            image_filename = os.path.basename(image_name)
            old_image_path = os.path.join(input_path, 'images', image_name + ".jpg")
            new_image_path = os.path.join(class_path, image_filename + ".jpg")
            shutil.copyfile(old_image_path, new_image_path)

# Example usage
folder_path = os.path.join(os.path.dirname(__file__), "Data", "food-101")
input_dataset_path = os.path.join(os.path.dirname(__file__), "Data", "food-101")
output_dataset_path = os.path.join(os.path.dirname(__file__), "Data", "food-101-ImageFolder")

transform_dataset(input_dataset_path, output_dataset_path)
