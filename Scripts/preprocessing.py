import torch
import torchvision
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

def get_data_loaders(train_dir, val_dir, test_dir=None, batch_size=32):
    # Define transformations for training and validation
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),          # Resize images
        transforms.RandomHorizontalFlip(),      # Data augmentation
        transforms.RandomRotation(20),         # Data augmentation
        transforms.ToTensor(),                  # Convert images to PyTorch tensors
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize using ImageNet stats
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),          # Resize images
        transforms.ToTensor(),                  # Convert images to PyTorch tensors
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize using ImageNet stats
    ])

    # Load datasets
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Print class names
    class_names = train_dataset.classes
    print("Classes:", class_names)

    return train_loader, val_loader, class_names;


def get_test_loader(test_dir, batch_size):
    """
    Load and preprocess the test dataset.

    Args:
        test_dir (str): Path to the test dataset.
        batch_size (int): Batch size for DataLoader.

    Returns:
        DataLoader: Test DataLoader.
    """
    # Define transformations for the test set
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Load the test dataset
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)

    # Create DataLoader
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return test_loader
