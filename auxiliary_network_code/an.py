# Import necessary libraries
import os
import numpy as np
from PIL import Image
from tqdm import tqdm

# PyTorch libraries
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

# Define constants
IMAGE_DIR = '/raid/yil708/stoat_data/auxiliary_network_pics/labelled_auxiliary_network_pics/labelled_auxiliary_network_pics/'
LABEL_FILE = '/raid/yil708/stoat_data/auxiliary_network_code/labelled_stoat.txt'
BATCH_SIZE = 32
NUM_EPOCHS = 10
NUM_CLASSES = 4
LEARNING_RATE = 0.001

# Custom dataset class
class StoatDataset(Dataset):
    def __init__(self, image_dir, label_file, transform=None):
        # Load labels
        self.image_dir = image_dir
        self.transform = transform
        with open(label_file, 'r') as f:
            lines = f.readlines()
        self.image_labels = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 2:
                self.image_labels.append((parts[0], int(parts[1])))

    def __len__(self):
        return len(self.image_labels)

    def __getitem__(self, idx):
        image_name, label = self.image_labels[idx]
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# Data transformations
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224
    transforms.ToTensor(),          # Convert images to tensors
    transforms.Normalize(           # Normalize images
        mean=[0.485, 0.456, 0.406], # Standard mean for ImageNet
        std=[0.229, 0.224, 0.225]   # Standard deviation for ImageNet
    )
])

# Create dataset and dataloader
dataset = StoatDataset(IMAGE_DIR, LABEL_FILE, transform=data_transforms)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Load pre-trained ResNet50 model
model = models.resnet50(pretrained=True)

# Modify the final layer to match the number of classes
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, NUM_CLASSES)

# Move model to GPU if available
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{NUM_EPOCHS}"):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()           # Zero the parameter gradients
        outputs = model(inputs)         # Forward pass
        loss = criterion(outputs, labels) # Compute loss
        loss.backward()                 # Backward pass
        optimizer.step()                # Update weights

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / train_size
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Training Loss: {epoch_loss:.4f}")

    # Validation loop
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}/{NUM_EPOCHS}"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels)
            total += labels.size(0)
    accuracy = correct.double() / total
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Validation Accuracy: {accuracy:.4f}")

# Save the trained model
torch.save(model.state_dict(), 'stoat_position_model.pth')
