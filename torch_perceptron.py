import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# Check GPU status

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

# Print additional GPU details if available
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    print(f"Memory Cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
else:
    print("GPU not detected, running on CPU.")

# ================================
# Step 1: Load MNIST Data from Local Files
# ================================
def load_mnist_images(filename):
    """Loads MNIST images from a local binary file (big-endian format)."""
    with open(filename, 'rb') as f:
        magic, num, rows, cols = np.frombuffer(f.read(16), dtype='>u4')  # Read as big-endian
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows * cols).astype(np.float32) / 255.0  # Normalize

def load_mnist_labels(filename):
    """Loads MNIST labels from a local binary file (big-endian format)."""
    with open(filename, 'rb') as f:
        magic, num = np.frombuffer(f.read(8), dtype='>u4')  # Read as big-endian
        return np.frombuffer(f.read(), dtype=np.uint8)  # Labels remain integers (0-9)

# ================================
# Step 2: Define Custom PyTorch Dataset
# ================================
class MNISTDataset(Dataset):
    def __init__(self, images_path, labels_path):
        self.images = torch.tensor(load_mnist_images(images_path), dtype=torch.float32)
        self.labels = torch.tensor(load_mnist_labels(labels_path), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

# ================================
# Step 3: Load Dataset from Local Files
# ================================
# Paths to local MNIST files
train_images_path = "train-images.idx3-ubyte"
train_labels_path = "train-labels.idx1-ubyte"
test_images_path = "t10k-images.idx3-ubyte"
test_labels_path = "t10k-labels.idx1-ubyte"

# Create dataset instances
train_dataset = MNISTDataset(train_images_path, train_labels_path)
test_dataset = MNISTDataset(test_images_path, test_labels_path)

# Create DataLoaders for batch processing
batch_size = 32
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# ================================
# Step 4: Define Neural Network Architecture
# ================================
class NeuralNetwork(nn.Module):
    def __init__(self, input_size=784, hidden_size1=128, hidden_size2=64, output_size=10):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size2, output_size)
    
    def forward(self, x):
        x = x.view(-1, 784)  # Flatten 28x28 to 784
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)  # No softmax (CrossEntropyLoss handles it)
        return x

# ================================
# Step 5: Define Loss Function & Optimizer
# ================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
model = NeuralNetwork().to(device)
criterion = nn.CrossEntropyLoss()  # Cross-entropy loss for classification
optimizer = optim.Adam(model.parameters(), lr=0.01)  # Adam optimizer

# ================================
# Step 6: Train the Model
# ================================
def train_model(model, train_loader, criterion, optimizer, epochs=10):
    model.train()  # Set model to training mode

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        accuracy = 100 * correct / total
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}, Accuracy: {accuracy:.2f}%")

# Train the model
train_model(model, train_loader, criterion, optimizer, epochs=10)

# ================================
# Step 7: Evaluate the Model
# ================================
def evaluate_model(model, test_loader, criterion):
    model.eval()  # Set model to evaluation mode
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    print(f"Test Loss: {total_loss / len(test_loader):.4f}, Test Accuracy: {accuracy:.2f}%")

# Evaluate the model
evaluate_model(model, test_loader, criterion)

# ================================
# Step 8: Hyperparameter Tuning
# ================================
def train_and_evaluate(learning_rate=0.01, hidden_size1=128, hidden_size2=64, batch_size=32, epochs=10):
    """
    Trains and evaluates the model with different hyperparameters.
    """
    print("\n=================================")
    print(f"Training with Hyperparameters:\n  Learning Rate: {learning_rate}\n  Hidden Layer 1 Neurons: {hidden_size1}\n  Hidden Layer 2 Neurons: {hidden_size2}\n  Batch Size: {batch_size}\n  Epochs: {epochs}")
    print("=================================")

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    model = NeuralNetwork(hidden_size1=hidden_size1, hidden_size2=hidden_size2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_model(model, train_loader, criterion, optimizer, epochs)
    evaluate_model(model, test_loader, criterion)

# Run experiments
train_and_evaluate(learning_rate=0.01, hidden_size1=256, hidden_size2=128, batch_size=64, epochs=10)