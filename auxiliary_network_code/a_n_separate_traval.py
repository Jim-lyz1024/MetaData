import wandb

def train():    
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

    # wandb library
    import wandb

    # Initialize wandb
    wandb.init(project='stoat_position_classification')

    # Define constants (You can use wandb.config for hyperparameters)
    config = wandb.config
    config.IMAGE_DIR = '/data/yil708/Meta_Data/MetaData/auxiliary_network_code/auxiliary_network_pics/cropped_labelled_auxiliary_network_pics/'
    config.TRAIN_LABEL_FILE = '/data/yil708/Meta_Data/MetaData/auxiliary_network_code/labelled_stoat_train.txt'
    config.VAL_LABEL_FILE = '/data/yil708/Meta_Data/MetaData/auxiliary_network_code/labelled_stoat_val.txt'
    config.TEST_LABEL_FILE = '/data/yil708/Meta_Data/MetaData/auxiliary_network_code/labelled_stoat_test.txt'

    # config.BATCH_SIZE = wandb.config.BATCH_SIZE
    # config.NUM_EPOCHS = wandb.config.NUM_EPOCHS
    config.NUM_CLASSES = 4
    # config.LEARNING_RATE = wandb.config.LEARNING_RATE

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
    train_dataset = StoatDataset(config.IMAGE_DIR, config.TRAIN_LABEL_FILE, transform=data_transforms)
    val_dataset = StoatDataset(config.IMAGE_DIR, config.VAL_LABEL_FILE, transform=data_transforms)
    test_dataset = StoatDataset(config.IMAGE_DIR, config.TEST_LABEL_FILE, transform=data_transforms)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # Load pre-trained ResNet50 model
    model = models.resnet50(pretrained=True)

    # Modify the final layer to match the number of classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, config.NUM_CLASSES)

    # Move model to GPU if available
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # Initialize lists to store metrics
    train_losses = []
    val_losses = []
    val_accuracies = []

    # Training loop
    for epoch in range(config.NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        for inputs, labels, _ in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{config.NUM_EPOCHS}"):
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
        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS}, Training Loss: {epoch_loss:.4f}")

        # Validation loop
        model.eval()
        val_running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels, _ in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}/{config.NUM_EPOCHS}"):
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
        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS}, Validation Loss: {val_epoch_loss:.4f}, Validation Accuracy: {accuracy:.4f}")

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
        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS}, Test Accuracy: {test_accuracy:.4f}")
        
        # Log metrics to wandb
        wandb.log({
            'epoch': epoch + 1,
            'train_loss': epoch_loss,
            'val_loss': val_epoch_loss,
            'val_accuracy': accuracy.item(),
            'test_accuracy': test_accuracy,
            'learning_rate': config.LEARNING_RATE
        })

    #######################################################################################
    # Define output directory (relative to the parent directory of the script)
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)  # Create the output folder if it doesn't exist

    # Save the trained model with epoch and batch size in filename inside the output folder
    torch.save(model.state_dict(), os.path.join(output_dir, f'stoat_position_model_{config.NUM_EPOCHS}_epochs_{config.BATCH_SIZE}_{config.LEARNING_RATE}_batch.pth'))

    # Plot and save Training Loss as PDF in the output folder
    plt.figure(figsize=(10,5))
    plt.title("Training Loss Over Epochs")
    plt.plot(range(1, config.NUM_EPOCHS+1), train_losses, label="Train Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"training_loss_{config.NUM_EPOCHS}_epochs_{config.BATCH_SIZE}_{config.LEARNING_RATE}_batch.pdf"))  # Save as PDF
    plt.close()  # Close the figure to avoid display

    # Plot and save Validation Loss as PDF in the output folder
    plt.figure(figsize=(10,5))
    plt.title("Validation Loss Over Epochs")
    plt.plot(range(1, config.NUM_EPOCHS+1), val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"validation_loss_{config.NUM_EPOCHS}_epochs_{config.BATCH_SIZE}_{config.LEARNING_RATE}_batch.pdf"))  # Save as PDF
    plt.close()  # Close the figure to avoid display

    # Plot and save Validation Accuracy as PDF in the output folder
    plt.figure(figsize=(10,5))
    plt.title("Validation Accuracy Over Epochs")
    plt.plot(range(1, config.NUM_EPOCHS+1), val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"validation_accuracy_{config.NUM_EPOCHS}_epochs_{config.BATCH_SIZE}_{config.LEARNING_RATE}_batch.pdf"))  # Save as PDF
    plt.close()  # Close the figure to avoid display

    # Log final training plots to wandb
    # wandb.log({"Training Loss": wandb.Image(os.path.join(output_dir, f"training_loss_{config.NUM_EPOCHS}_epochs_{config.BATCH_SIZE}_{config.LEARNING_RATE}_batch.pdf"))})
    # wandb.log({"Validation Loss": wandb.Image(os.path.join(output_dir, f"validation_loss_{config.NUM_EPOCHS}_epochs_{config.BATCH_SIZE}_{config.LEARNING_RATE}_batch.pdf"))})
    # wandb.log({"Validation Accuracy": wandb.Image(os.path.join(output_dir, f"validation_accuracy_{config.NUM_EPOCHS}_epochs_{config.BATCH_SIZE}_{config.LEARNING_RATE}_batch.pdf"))})

    # Viewing Specific Model Predictions and Computing Confusion Matrix on Validation Set
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

    # Save predictions CSV in the output folder
    results_df.to_csv(os.path.join(output_dir, f'model_predictions_{config.NUM_EPOCHS}_epochs_{config.BATCH_SIZE}_{config.LEARNING_RATE}_batch.csv'), index=False)

    # Compute confusion matrix on validation set
    cm = confusion_matrix(all_labels, all_preds)

    # Plot confusion matrix and save as PDF in the output folder
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[label_mapping[i] for i in range(config.NUM_CLASSES)],
                yticklabels=[label_mapping[i] for i in range(config.NUM_CLASSES)])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix on Validation Set')
    plt.savefig(os.path.join(output_dir, f"confusion_matrix_{config.NUM_EPOCHS}_epochs_{config.BATCH_SIZE}_{config.LEARNING_RATE}_batch.pdf"))  # Save as PDF
    plt.close()  # Close the figure to avoid display

    # Log confusion matrix to wandb
    # wandb.log({"Confusion Matrix": wandb.Image(os.path.join(output_dir, f"confusion_matrix_{config.NUM_EPOCHS}_epochs_{config.BATCH_SIZE}_{config.LEARNING_RATE}_batch.pdf"))})

    # Print classification report
    print("Classification Report on Validation Set:")
    print(classification_report(all_labels, all_preds, target_names=[label_mapping[i] for i in range(config.NUM_CLASSES)]))

    # Log classification report to wandb
    report = classification_report(all_labels, all_preds, target_names=[label_mapping[i] for i in range(config.NUM_CLASSES)], output_dict=True)
    # wandb.log({"Validation Classification Report": report})

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
    test_accuracy_final = test_correct / test_total
    print(f"Test Accuracy: {test_accuracy_final:.4f}")

    # Log test accuracy to wandb
    wandb.log({'test_accuracy_final': test_accuracy_final})

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
    test_results_df.to_csv(os.path.join(output_dir, f'test_predictions_{config.NUM_EPOCHS}_epochs_{config.BATCH_SIZE}_{config.LEARNING_RATE}_batch.csv'), index=False)

    # Compute confusion matrix on test set
    test_cm = confusion_matrix(test_labels, test_preds)

    # Plot confusion matrix for test set and save as PDF in the output folder
    plt.figure(figsize=(8,6))
    sns.heatmap(test_cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[label_mapping[i] for i in range(config.NUM_CLASSES)],
                yticklabels=[label_mapping[i] for i in range(config.NUM_CLASSES)])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix on Test Set')
    plt.savefig(os.path.join(output_dir, f"test_confusion_matrix_{config.NUM_EPOCHS}_epochs_{config.BATCH_SIZE}_{config.LEARNING_RATE}_batch.pdf"))  # Save as PDF
    plt.close()  # Close the figure to avoid display

    # Log test confusion matrix to wandb
    # wandb.log({"Test Confusion Matrix": wandb.Image(os.path.join(output_dir, f"test_confusion_matrix_{config.NUM_EPOCHS}_epochs_{config.BATCH_SIZE}_{config.LEARNING_RATE}_batch.pdf"))})

    # Print classification report for test set
    print("Classification Report on Test Set:")
    print(classification_report(test_labels, test_preds, target_names=[label_mapping[i] for i in range(config.NUM_CLASSES)]))

    # Log test classification report to wandb
    test_report = classification_report(test_labels, test_preds, target_names=[label_mapping[i] for i in range(config.NUM_CLASSES)], output_dict=True)
    # wandb.log({"Test Classification Report": test_report})

    # Finish the wandb run
    wandb.finish()


sweep_config = {
    'method': 'grid',  # Can be 'grid', 'random', or 'bayes'
    'metric': {
        'name': 'test_accuracy_final',
        'goal': 'maximize'  # Use 'minimize' if optimizing loss
    },
    'parameters': {
        'BATCH_SIZE': {
            'values': [32, 64, 128]
        },
        'LEARNING_RATE': {
            'values': [0.01, 1e-3, 1e-4, 1e-5]
        },
        'NUM_EPOCHS': {
            'values': [10, 50, 75]
        }
    }
}

sweep_id = wandb.sweep(sweep_config, project='stoat_position_classification')
wandb.agent(sweep_id, function=train)