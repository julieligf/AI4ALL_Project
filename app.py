import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the model architecture
model = models.resnet50()
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, 4)  # Adjust for your number of classes

# Load the trained weights
model.load_state_dict(torch.load('resnet50_model.pth', map_location=device))
model.to(device)
model.eval()

# Define class names (replace with your actual class names)
class_names = ['dry', 'wet', 'foggy', 'snowy']  # Replace with your class names

# Define the image preprocessing steps
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # Normalize with the same mean and std as during training
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # Standard for ResNet models
        std=[0.229, 0.224, 0.225]
    )
])

# Streamlit App
st.title("Weather Condition Classification")
st.write("Upload an image to get the weather condition prediction.")

uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Load the image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    input_tensor = preprocess(image).unsqueeze(0).to(device)

    # Make prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        # Get the top prediction
        top_index = np.argmax(probabilities)
        top_class = class_names[top_index]
        top_probability = probabilities[top_index]

    # Display the results
    st.write(f"**Predicted Weather Condition:** {top_class}")
    st.write(f"**Confidence:** {top_probability:.4f}")

    # Display probabilities for all classes
    st.write("### Class Probabilities:")
    for class_name, probability in zip(class_names, probabilities):
        st.write(f"{class_name}: {probability:.4f}")
