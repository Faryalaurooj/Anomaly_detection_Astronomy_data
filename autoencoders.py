import os
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim

# Initialize the CLIP processor and model
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

# Path to the folder containing images
image_folder = "/home/faryal/Downloads/Anomaly detection/space-images-category/versions/1/space_images/constellation/"

# Initialize a list to store features
all_features = []
image_filenames = []

# Extract CLIP features from images
for filename in os.listdir(image_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        image_path = os.path.join(image_folder, filename)
        
        # Load the image
        image = Image.open(image_path).convert("RGB")
        
        # Preprocess and extract features
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model.get_image_features(**inputs)
        
        # Convert the output tensor to a NumPy array and store it
        features = outputs.cpu().numpy()
        all_features.append(features)
        image_filenames.append(filename)
        print(f"Extracted features for {filename}")

# Convert features to a NumPy array
all_features = np.concatenate(all_features, axis=0)

# Normalize features
scaler = StandardScaler()
normalized_features = scaler.fit_transform(all_features)

# Split data into train and test sets
train_data, test_data = train_test_split(normalized_features, test_size=0.2, random_state=42)

# Convert train and test sets to PyTorch tensors
train_data = torch.tensor(train_data, dtype=torch.float32)
test_data = torch.tensor(test_data, dtype=torch.float32)

# Define the Autoencoder model
class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, encoding_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid()  # Sigmoid for normalized features
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Instantiate the autoencoder
input_dim = train_data.shape[1]
encoding_dim = 64  # Adjust as needed
autoencoder = Autoencoder(input_dim, encoding_dim)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

# Train the autoencoder
epochs = 50
batch_size = 32
autoencoder.train()
for epoch in range(epochs):
    for i in range(0, train_data.size(0), batch_size):
        batch = train_data[i:i + batch_size]
        optimizer.zero_grad()
        reconstructed = autoencoder(batch)
        loss = criterion(reconstructed, batch)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

# Evaluate the autoencoder on test data
autoencoder.eval()
reconstructed_test = autoencoder(test_data)
reconstruction_error = ((test_data - reconstructed_test) ** 2).mean(dim=1).detach().numpy()

# Determine anomalies based on reconstruction error
threshold = np.percentile(reconstruction_error, 95)  # Top 5% as anomalies
anomaly_labels = (reconstruction_error > threshold).astype(int)  # 1 for anomaly, 0 for normal

# Save results to a CSV file
output_file = "/home/faryal/Downloads/Anomaly detection/Autoencoder_Anomaly_Detection_Results.csv"
with open(output_file, mode="w", newline="") as file:
    import csv
    writer = csv.writer(file)
    writer.writerow(["Filename", "Reconstruction Error", "Anomaly"])
    for i, filename in enumerate(image_filenames):
        if i < len(anomaly_labels):  # To match test data size
            writer.writerow([filename, reconstruction_error[i], "Anomaly" if anomaly_labels[i] == 1 else "Normal"])

print(f"Results saved to {output_file}")

