import os
import csv
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Initialize the CLIP processor and model
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

# Path to the folder containing images
image_folder = "/home/faryal/Downloads/Anomaly detection/space-images-category/versions/1/space_images/constellation/"

# Output file for saving results
output_file = "/home/faryal/Downloads/Anomaly detection/Anomaly_detection_results.csv"

# Initialize a list to store features
all_features = []
image_filenames = []  # To keep track of filenames

# Loop through all images in the folder
for filename in os.listdir(image_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        image_path = os.path.join(image_folder, filename)
        
        # Load the image
        image = Image.open(image_path).convert("RGB")
        
        # Preprocess and extract features
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():  # Disable gradient calculation for inference
            outputs = model.get_image_features(**inputs)
        
        # Convert the output tensor to a NumPy array and store it
        features = outputs.cpu().numpy()
        all_features.append(features)
        image_filenames.append(filename)  # Save the filename
        
        print(f"Extracted features for {filename}")

# Convert list of features to a NumPy array (shape: num_images x feature_size)
all_features = np.concatenate(all_features, axis=0)

# **Step 1: Normalize the extracted features**
scaler = StandardScaler()
normalized_features = scaler.fit_transform(all_features)

# **Step 2: Apply Isolation Forest for anomaly detection**
clf = IsolationForest(contamination=0.05)  # Adjust contamination based on the dataset
clf.fit(normalized_features)

# **Step 3: Get anomaly scores**
anomaly_scores = clf.decision_function(normalized_features)  # Higher is less anomalous
predictions = clf.predict(normalized_features)  # -1 for anomalies, 1 for normal

# **Step 4: Save results to a CSV file**
with open(output_file, mode="w", newline="") as file:
    writer = csv.writer(file)
    # Write header
    writer.writerow(["Filename", "Anomaly Score", "Prediction"])
    # Write data rows
    for i, filename in enumerate(image_filenames):
        status = "Anomaly" if predictions[i] == -1 else "Normal"
        writer.writerow([filename, anomaly_scores[i], status])

print(f"Results saved to {output_file}")

