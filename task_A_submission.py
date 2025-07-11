import os 
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from efficientnet_pytorch import EfficientNet
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Set the root path for the dataset
data_dir = '/content/drive/MyDrive/Task_A'

# Define paths to training and validation folders
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')

# Set device to GPU if available, otherwise fall back to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image size expected by EfficientNet
input_size = 224

# Define data augmentation and normalization for training set
train_transform = transforms.Compose([
    transforms.Resize((input_size, input_size)),              # Resize images
    transforms.RandomHorizontalFlip(),                        # Randomly flip images horizontally
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Random color jitter
    transforms.RandomRotation(10),                            # Random rotation within 10 degrees
    transforms.ToTensor(),                                    # Convert images to PyTorch tensors
    transforms.Normalize([0.485, 0.456, 0.406],                # Normalize using ImageNet mean
                         [0.229, 0.224, 0.225])                # and std deviation
])

# Define validation preprocessing (no augmentation, only resizing and normalization)
val_transform = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Load training and validation datasets
train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

# Create data loaders for batching and shuffling
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Load pre-trained EfficientNet-B0 and replace final layer for 2-class classification
model = EfficientNet.from_pretrained('efficientnet-b0')
model._fc = nn.Linear(model._fc.in_features, 2)
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=2e-4)

# Cosine annealing scheduler to adjust learning rate over epochs
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)

# Initialize tracking variables for best validation accuracy and final predictions
best_val_acc = 0.0
final_train_preds = []
final_train_labels = []
final_val_preds = []
final_val_labels = []

# Training and validation loop for 6 epochs
for epoch in range(1, 7):
    model.train()
    all_train_preds, all_train_labels = [], []

    # Train over all batches
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()               # Clear gradients
        outputs = model(images)             # Forward pass
        loss = criterion(outputs, labels)   # Compute loss
        loss.backward()                     # Backpropagation
        optimizer.step()                    # Update weights

        preds = torch.argmax(outputs, 1)    # Get predicted classes
        all_train_preds.extend(preds.cpu().numpy())
        all_train_labels.extend(labels.cpu().numpy())

    train_acc = accuracy_score(all_train_labels, all_train_preds)
    scheduler.step()  # Update learning rate

    # Evaluation mode (no gradient tracking)
    model.eval()
    all_val_preds, all_val_labels = [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, 1)
            all_val_preds.extend(preds.cpu().numpy())
            all_val_labels.extend(labels.cpu().numpy())

    val_acc = accuracy_score(all_val_labels, all_val_preds)
    print(f"Epoch {epoch} done | Training Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    # Save the model if it has the best validation accuracy so far
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_gender_model.pt")

    # Save predictions from the final epoch for metric reporting
    final_train_preds = all_train_preds
    final_train_labels = all_train_labels
    final_val_preds = all_val_preds
    final_val_labels = all_val_labels

# Compute final training metrics
train_precision, train_recall, train_f1, _ = precision_recall_fscore_support(
    final_train_labels, final_train_preds, average='binary')
train_acc = accuracy_score(final_train_labels, final_train_preds)

# Compute final validation metrics
val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(
    final_val_labels, final_val_preds, average='binary')
val_acc = accuracy_score(final_val_labels, final_val_preds)

# Print final training results
print("\nFinal Training Scores:")
print(f"Accuracy : {train_acc:.4f}")
print(f"Precision: {train_precision:.4f}")
print(f"Recall   : {train_recall:.4f}")
print(f"F1 Score : {train_f1:.4f}")

# Print final validation results
print("\nFinal Validation Scores:")
print(f"Accuracy : {val_acc:.4f}")
print(f"Precision: {val_precision:.4f}")
print(f"Recall   : {val_recall:.4f}")
print(f"F1 Score : {val_f1:.4f}")
