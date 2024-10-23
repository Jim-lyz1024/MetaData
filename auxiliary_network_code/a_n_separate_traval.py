# Import necessary libraries
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# PyTorch libraries
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

# Sklearn libraries
from sklearn.metrics import confusion_matrix, classification_report

# Define constants
IMAGE_DIR = '/data/yil708/Meta_Data/MetaData/auxiliary_network_code/auxiliary_network_pics/cropped_labelled_auxiliary_network_pics/'
TRAIN_LABEL_FILE = '/data/yil708/Meta_Data/MetaData/auxiliary_network_code/labelled_stoat_train.txt'
VAL_LABEL_FILE = '/data/yil708/Meta_Data/MetaData/auxiliary_network_code/labelled_stoat_val.txt'
TEST_LABEL_FILE = '/data/yil708/Meta_Data/MetaData/auxiliary_network_code/labelled_stoat_test.txt'

BATCH_SIZE = 64
NUM_EPOCHS = 2
# NUM_EPOCHS = 50
NUM_CLASSES = 4
# LEARNING_RATE = 0.001
LEARNING_RATE = 0.0001

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
        return image, label, image_name  # Return image name for later use

# Data transformations
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224
    transforms.RandomHorizontalFlip(),   # Data augmentation
    transforms.RandomRotation(10),
    transforms.ToTensor(),          # Convert images to tensors
    transforms.Normalize(           # Normalize images
        mean=[0.485, 0.456, 0.406], # Standard mean for ImageNet
        std=[0.229, 0.224, 0.225]   # Standard deviation for ImageNet
    )
])

# Create datasets and dataloaders
train_dataset = StoatDataset(IMAGE_DIR, TRAIN_LABEL_FILE, transform=data_transforms)
val_dataset = StoatDataset(IMAGE_DIR, VAL_LABEL_FILE, transform=data_transforms)
test_dataset = StoatDataset(IMAGE_DIR, TEST_LABEL_FILE, transform=data_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Load pre-trained ResNet50 model
model = models.resnet50(pretrained=True)

# Modify the final layer to match the number of classes
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, NUM_CLASSES)

# Move model to GPU if available
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Initialize lists to store metrics
train_losses = []
val_losses = []
val_accuracies = []

# Training loop
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    for inputs, labels, _ in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{NUM_EPOCHS}"):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()           # Zero the parameter gradients
        outputs = model(inputs)         # Forward pass
        loss = criterion(outputs, labels) # Compute loss
        loss.backward()                 # Backward pass
        optimizer.step()                # Update weights

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_dataset)
    train_losses.append(epoch_loss)
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Training Loss: {epoch_loss:.4f}")

    # Validation loop
    model.eval()
    val_running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels, _ in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}/{NUM_EPOCHS}"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)  # Compute validation loss
            val_running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels)
            total += labels.size(0)
    val_epoch_loss = val_running_loss / len(val_dataset)  # Compute average validation loss
    val_losses.append(val_epoch_loss)
    accuracy = correct.double() / total
    val_accuracies.append(accuracy.item())
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Validation Loss: {val_epoch_loss:.4f}, Validation Accuracy: {accuracy:.4f}")

#######################################################################################
# Define output directory (relative to the parent directory of the script)
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(output_dir, exist_ok=True)  # Create the output folder if it doesn't exist

# Save the trained model with epoch and batch size in filename inside the output folder
torch.save(model.state_dict(), os.path.join(output_dir, f'stoat_position_model_{NUM_EPOCHS}_epochs_{BATCH_SIZE}_{LEARNING_RATE}_batch.pth'))

# Plot and save Training Loss as PDF in the output folder
plt.figure(figsize=(10,5))
plt.title("Training Loss Over Epochs")
plt.plot(range(1, NUM_EPOCHS+1), train_losses, label="Train Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig(os.path.join(output_dir, f"training_loss_{NUM_EPOCHS}_epochs_{BATCH_SIZE}_{LEARNING_RATE}_batch.pdf"))  # Save as PDF
plt.close()  # Close the figure to avoid display

# Plot and save Validation Loss as PDF in the output folder
plt.figure(figsize=(10,5))
plt.title("Validation Loss Over Epochs")
plt.plot(range(1, NUM_EPOCHS+1), val_losses, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig(os.path.join(output_dir, f"validation_loss_{NUM_EPOCHS}_epochs_{BATCH_SIZE}_{LEARNING_RATE}_batch.pdf"))  # Save as PDF
plt.close()  # Close the figure to avoid display

# Plot and save Validation Accuracy as PDF in the output folder
plt.figure(figsize=(10,5))
plt.title("Validation Accuracy Over Epochs")
plt.plot(range(1, NUM_EPOCHS+1), val_accuracies, label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig(os.path.join(output_dir, f"validation_accuracy_{NUM_EPOCHS}_epochs_{BATCH_SIZE}_{LEARNING_RATE}_batch.pdf"))  # Save as PDF
plt.close()  # Close the figure to avoid display

# Viewing Specific Model Predictions and Computing Confusion Matrix
# Collect all predictions and true labels
all_preds = []
all_labels = []
all_image_names = []

model.eval()
with torch.no_grad():
    for inputs, labels, image_names in tqdm(val_loader, desc="Generating Predictions on Validation Set"):
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_image_names.extend(image_names)

# Create a DataFrame to store results
results_df = pd.DataFrame({
    'Image': all_image_names,
    'True Label': all_labels,
    'Predicted Label': all_preds
})

# Map numeric labels to actual position labels
label_mapping = {0: 'Front', 1: 'Back', 2: 'Left', 3: 'Right'}
results_df['True Label Name'] = results_df['True Label'].map(label_mapping)
results_df['Predicted Label Name'] = results_df['Predicted Label'].map(label_mapping)

# Display the first few predictions
print(results_df.head())

# Save predictions CSV in the output folder
results_df.to_csv(os.path.join(output_dir, f'model_predictions_{NUM_EPOCHS}_epochs_{BATCH_SIZE}_{LEARNING_RATE}_batch.csv'), index=False)

# # Compute confusion matrix
# cm = confusion_matrix(all_labels, all_preds)

# # Plot confusion matrix and save as PDF in the output folder
# plt.figure(figsize=(8,6))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#             xticklabels=[label_mapping[i] for i in range(NUM_CLASSES)],
#             yticklabels=[label_mapping[i] for i in range(NUM_CLASSES)])
# plt.ylabel('True Label')
# plt.xlabel('Predicted Label')
# plt.title('Confusion Matrix')
# plt.savefig(os.path.join(output_dir, f"confusion_matrix_{NUM_EPOCHS}_epochs_{BATCH_SIZE}_{LEARNING_RATE}_batch.pdf"))  # Save as PDF
# plt.close()  # Close the figure to avoid display

# Print classification report
# print("Classification Report:")
# print(classification_report(all_labels, all_preds, target_names=[label_mapping[i] for i in range(NUM_CLASSES)]))

#############################################################################################
# =========================================
# Evaluate the Model on the Test Dataset
# =========================================

# Collect all predictions and true labels on the test set
test_preds = []
test_labels = []
test_image_names = []

model.eval()
with torch.no_grad():
    for inputs, labels, image_names in tqdm(test_loader, desc="Evaluating on Test Dataset"):
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        test_preds.extend(preds.cpu().numpy())
        test_labels.extend(labels.cpu().numpy())
        test_image_names.extend(image_names)

# Compute test accuracy
test_correct = np.sum(np.array(test_preds) == np.array(test_labels))
test_total = len(test_labels)
test_accuracy = test_correct / test_total
print(f"Test Accuracy: {test_accuracy:.4f}")

# Create a DataFrame to store test results
test_results_df = pd.DataFrame({
    'Image': test_image_names,
    'True Label': test_labels,
    'Predicted Label': test_preds
})

# Map numeric labels to actual position labels
test_results_df['True Label Name'] = test_results_df['True Label'].map(label_mapping)
test_results_df['Predicted Label Name'] = test_results_df['Predicted Label'].map(label_mapping)

# Save test predictions CSV in the output folder
test_results_df.to_csv(os.path.join(output_dir, f'test_predictions_{NUM_EPOCHS}_epochs_{BATCH_SIZE}_{LEARNING_RATE}_batch.csv'), index=False)

# Compute confusion matrix on test set
test_cm = confusion_matrix(test_labels, test_preds)

# Plot confusion matrix for test set and save as PDF in the output folder
plt.figure(figsize=(8,6))
sns.heatmap(test_cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=[label_mapping[i] for i in range(NUM_CLASSES)],
            yticklabels=[label_mapping[i] for i in range(NUM_CLASSES)])
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix on Test Set')
plt.savefig(os.path.join(output_dir, f"test_confusion_matrix_{NUM_EPOCHS}_epochs_{BATCH_SIZE}_{LEARNING_RATE}_batch.pdf"))  # Save as PDF
plt.close()  # Close the figure to avoid display

# Print classification report for test set
print("Classification Report on Test Set:")
print(classification_report(test_labels, test_preds, target_names=[label_mapping[i] for i in range(NUM_CLASSES)]))