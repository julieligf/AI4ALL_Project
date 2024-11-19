import torch
from torchvision import models
from preprocessing import get_data_loaders, get_test_loader
from torchvision.models import ResNet50_Weights

# Load DataLoaders
test_dir = "/Users/mariocortez/Downloads/AI4ALL_Project/dataset/test"
test_loader = get_test_loader(test_dir, batch_size=32)

# Load the model
model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, 4)
model.load_state_dict(torch.load("resnet50_model.pth", weights_only=True))
model.eval()

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Evaluate the model
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

print(f"Test Accuracy: {correct/total*100:.2f}%")
# Compare this snippet from Scripts/evaluate.py: