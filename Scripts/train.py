import torch
import os
from torch import nn, optim
from torchvision import models
from preprocessing import get_data_loaders
from torchvision.models import ResNet50_Weights
# Load DataLoaders
base_dir = os.path.dirname(os.path.abspath(__file__))
train_dir = os.path.join(base_dir, "../dataset/train")
val_dir = os.path.join(base_dir, "../dataset/validation")

train_loader, val_loader, _ = get_data_loaders(train_dir, val_dir, batch_size=32)

# Build the model


model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
for param in model.parameters():
    param.requires_grad = False  # Freeze layers
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 4)  # 4 output classes

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")

# Save the trained model
torch.save(model.state_dict(), "resnet50_model.pth")
