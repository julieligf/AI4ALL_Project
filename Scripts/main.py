# main.py
import logging
from image_conversion import load_and_convert_images
from preprocessing import preprocess_images
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)

def main():
    folder_path = 'Images'  # Path to your images folder
    logging.info("Loading and converting images from %s", folder_path)

    # Load and convert images
    features, labels = load_and_convert_images(folder_path)
    if features is None or labels is None:
        logging.error("Failed to load images. Exiting.")
        return

    # Preprocess the images
    logging.info("Preprocessing images")
    features = preprocess_images(features)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Create and train the decision tree model
    model = DecisionTreeClassifier()
    logging.info("Training the decision tree model")
    model.fit(X_train, y_train)

    # Evaluate the model
    accuracy = model.score(X_test, y_test)
    logging.info(f"Model accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    main()
