import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense # type: ignore
import os

print("All imports were successful!")
# Set up the image data generators for training
train_dir = 'train_data'

train_datagen = ImageDataGenerator(
    rescale=1./255,                # Normalize images
    rotation_range=20,            # Random rotations
    width_shift_range=0.2,        # Random width shifts
    height_shift_range=0.2,       # Random height shifts
    shear_range=0.2,              # Random shear
    zoom_range=0.2,               # Random zoom
    horizontal_flip=True,         # Random horizontal flips
    fill_mode='nearest'           # Fill missing pixels after transformations
)

train_generator = train_datagen.flow_from_directory(
    train_dir,                    # Directory containing your training images
    target_size=(150, 150),       # Resize images to 150x150
    batch_size=32,                # Number of images per batch
    class_mode='categorical'      # We have multiple categories (dry, wet, foggy, rainy)
)

# Build the deep learning model
model = Sequential()

# Convolutional Layer 1
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(MaxPooling2D((2, 2)))

# Convolutional Layer 2
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

# Convolutional Layer 3
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

# Flatten the output for the fully connected layers
model.add(Flatten())

# Fully Connected Layer
model.add(Dense(512, activation='relu'))

# Output Layer
model.add(Dense(4, activation='softmax'))  # 4 categories (dry, wet, foggy, rainy)

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Train the model
model.fit(train_generator, epochs=10, steps_per_epoch=100)

# Save the model
model.save('model/weather_model.h5')
