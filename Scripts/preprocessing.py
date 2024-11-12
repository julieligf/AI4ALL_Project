import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore

def preprocess_images():
    # Apply ImageDataGenerator to preprocess images
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Directory where images are stored
    dataset_dir = 'dataset/'

    # For training
    train_generator = datagen.flow_from_directory(
        os.path.join(dataset_dir, 'train'),
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical'
    )

    # For validation
    validation_generator = datagen.flow_from_directory(
        os.path.join(dataset_dir, 'validation'),
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical'
    )

    return train_generator, validation_generator
