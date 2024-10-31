# preprocessing.py
import numpy as np

def preprocess_images(features, target_size=(64, 64)):
    """
    Preprocess the images by normalizing pixel values and resizing them to a consistent size.

    Args:
        features (numpy.ndarray): Array of image features (flattened pixel values).
        target_size (tuple): Desired size for resizing the images.

    Returns:
        numpy.ndarray: Preprocessed image features.
    """
    # Normalize pixel values to the range [0, 1]
    features = features.astype('float32') / 255.0  # Normalize

    # Resize images to a consistent size (if applicable)
    # Note: This example assumes features are still in the original format.
    # You may want to resize before flattening in `image_conversion.py`.
    # Use cv2.resize() if handling images directly.

    return features
