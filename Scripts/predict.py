import numpy as np
from tensorflow.keras.preprocessing import image # type: ignore
from tensorflow.keras.models import load_model # type: ignore

# Load the trained model
model = load_model('weather_classifier_model/weather_classifier.h5')

# Path to the image you want to classify
img_path = 'path_to_new_image.jpg'

# Load and preprocess the image
img = image.load_img(img_path, target_size=(150, 150))
img_array = image.img_to_array(img) / 255.0  # Normalize image
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

# Predict the weather condition
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions, axis=1)

# Map predicted class index to label
labels = ['dry', 'wet', 'foggy', 'wet']
predicted_label = labels[predicted_class[0]]
print(f"Predicted weather condition: {predicted_label}")
