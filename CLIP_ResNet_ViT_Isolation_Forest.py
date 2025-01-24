#This code will extract features from ResNet, ViT, and CLIP, combine them, and then use an Isolation Forest to detect anomalies.

import os
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from transformers import ViTImageProcessor, ViTModel
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import csv

# Paths and configurations
image_folder = "/home/faryal/Downloads/Anomaly detection/space-images-category/versions/1/space_images/cosmos_space"  # Update with your image folder
output_file = "/home/faryal/Downloads/Anomaly detection/anomaly_detection_results_combined_models.csv"  # Update with your output path

# Initialize ResNet model (pre-trained)
weights = ResNet50_Weights.IMAGENET1K_V1
resnet = resnet50(weights=weights)
resnet.fc = torch.nn.Identity()  # Remove the classification head for feature extraction
resnet.eval()

# Initialize ViT model (pre-trained)
vit_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
vit_model.eval()

# Image transformation for ResNet
resnet_transform = weights.transforms()  # Use pre-defined transforms for ResNet

# Function to extract features from ResNet
def extract_resnet_features(image):
    image_tensor = resnet_transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        resnet_features = resnet(image_tensor)
    return resnet_features.cpu().numpy().flatten()

# Function to extract features from ViT
def extract_vit_features(image):
    inputs = vit_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        vit_features = vit_model(**inputs).last_hidden_state[:, 0, :]  # Use CLS token features
    return vit_features.cpu().numpy().flatten()

# Loop through images and extract combined features
combined_features = []
image_filenames = []

if not os.path.exists(image_folder):
    raise FileNotFoundError(f"Image folder does not exist: {image_folder}")

for filename in os.listdir(image_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        image_path = os.path.join(image_folder, filename)
        image = Image.open(image_path).convert("RGB")

        # Extract features from ResNet and ViT
        resnet_feat = extract_resnet_features(image)
        vit_feat = extract_vit_features(image)

        # Combine features
        combined_feat = np.concatenate([resnet_feat, vit_feat])
        combined_features.append(combined_feat)
        image_filenames.append(filename)
        print(f"Extracted combined features for {filename}")

# Convert features to NumPy array
combined_features = np.array(combined_features)

# Normalize features
scaler = StandardScaler()
normalized_features = scaler.fit_transform(combined_features)

# Train Isolation Forest
clf = IsolationForest(contamination=0.05, random_state=42)
clf.fit(normalized_features)

# Predict anomalies
anomaly_scores = clf.decision_function(normalized_features)  # Higher is less anomalous
predictions = clf.predict(normalized_features)  # -1 for anomalies, 1 for normal

# Save results to a CSV file
with open(output_file, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Filename", "Anomaly Score", "Prediction"])
    for i, filename in enumerate(image_filenames):
        writer.writerow([filename, anomaly_scores[i], "Anomaly" if predictions[i] == -1 else "Normal"])

print(f"Anomaly detection results saved to {output_file}")

