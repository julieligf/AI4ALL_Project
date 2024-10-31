import os
import cv2
import numpy as np

def load_images_from_folder(folder):
    """
    Load all images from the folder.

    Args:
        folder (str): The path to the folder containing images.

    Returns:
        list: A list of loaded images as NumPy arrays.
    """
    images = []
    for filename in os.listdir(folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):  # Adjust based on your image format
            img_path = os.path.join(folder, filename)
            image = cv2.imread(img_path)

            if image is not None:
                images.append(image)
            else:
                print(f"Warning: Failed to load image at {img_path}")
    return images

def preprocess_images(images):
    """
    Preprocess images by resizing and normalizing.

    Args:
        images (list): List of images as NumPy arrays.

    Returns:
        list: List of processed images.
    """
    processed_images = []
    for image in images:
        # Resize image to a fixed size (e.g., 128x128)
        resized_image = cv2.resize(image, (128, 128))
        # Normalize pixel values to the range [0, 1]
        normalized_image = resized_image / 255.0
        processed_images.append(normalized_image)
    return processed_images

def calculate_megapixels(image):
    """
    Calculate the megapixels of an image.

    Args:
        image (ndarray): The image as a NumPy array.

    Returns:
        float: Megapixels of the image.
    """
    height, width, _ = image.shape
    megapixels = (height * width) / 1_000_000  # Convert to megapixels
    return megapixels

def load_and_convert_images(base_folder):
    """
    Load images from the given folder structure and convert them to a suitable format for machine learning.

    Args:
        base_folder (str): The path to the base folder containing weather condition folders.

    Returns:
        tuple: A tuple containing processed images and corresponding labels.
    """
    weather_conditions = ['wet', 'dry', 'foggy', 'icy']
    all_images = []
    labels = []

    for condition in weather_conditions:
        condition_folder = os.path.join(base_folder, condition)  # Path to the condition folder
        images = load_images_from_folder(condition_folder)  # Load images
        processed_images = preprocess_images(images)  # Preprocess images
        
        all_images.extend(processed_images)  # Add processed images to the list
        labels.extend([condition] * len(processed_images))  # Add labels for the condition

        # Calculate and print megapixels for the first processed image
        if processed_images:
            megapixels = calculate_megapixels(processed_images[0])
            print(f"First processed image for '{condition}' has {megapixels:.2f} megapixels.")

    return np.array(all_images), np.array(labels)  # Return as NumPy arrays for compatibility with ML models

def main():
    base_folder = './Images'  # Adjust this path if needed
    all_images, labels = load_and_convert_images(base_folder)  # Load and convert images
    print(f"Total images loaded: {len(all_images)}")
    print(f"Labels: {set(labels)}")

if __name__ == "__main__":
    main()
