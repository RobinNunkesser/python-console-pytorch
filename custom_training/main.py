import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Model definition
def get_model():
    model = nn.Sequential(
        nn.Linear(784, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )
    return model

model = get_model().to(device)
print(model)

# Optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# Prepare dataset
batch_size = 32
(x_train, y_train), (x_test, y_test) = nn.Module.__dict__.get('load_data', 
    lambda: (__import__('torchvision').datasets.MNIST(root='./data', train=True, download=True),
             __import__('torchvision').datasets.MNIST(root='./data', train=False, download=True)))()

# Alternative: Use Keras API for quick dataset loading
from torchvision import datasets, transforms

train_dataset = datasets.MNIST(root='./data', train=True, 
                               transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='./data', train=False,
                              transform=transforms.ToTensor(), download=True)

x_train = train_dataset.data.float().reshape(-1, 784).to(device) / 255.0
y_train = train_dataset.targets.to(device)
x_test = test_dataset.data.float().reshape(-1, 784).to(device) / 255.0
y_test = test_dataset.targets.to(device)

# Reserve 10,000 samples for validation
x_val = x_train[-10000:]
y_val = y_train[-10000:]
x_train = x_train[:-10000]
y_train = y_train[:-10000]

print(f"Training samples: {x_train.shape[0]}")
print(f"Validation samples: {x_val.shape[0]}")
print(f"Test samples: {x_test.shape[0]}")

# Training loop
epochs = 3
for epoch in range(epochs):
    print(f"\nStart of epoch {epoch}")
    
    # Shuffle training data
    indices = torch.randperm(x_train.shape[0])
    x_train_shuffled = x_train[indices]
    y_train_shuffled = y_train[indices]
    
    # Iterate over batches
    for step in range(0, x_train.shape[0], batch_size):
        x_batch = x_train_shuffled[step:step + batch_size]
        y_batch = y_train_shuffled[step:step + batch_size]
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(x_batch)
        loss_value = loss_fn(logits, y_batch)
        
        # Backward pass
        loss_value.backward()
        optimizer.step()
        
        if step % (100 * batch_size) == 0:
            print(f"Training loss (for 1 batch) at step {step}: {loss_value.item():.4f}")
            print(f"Seen so far: {step + x_batch.shape[0]} samples")

# Evaluation on validation set
model.eval()
with torch.no_grad():
    val_logits = model(x_val)
    val_loss = loss_fn(val_logits, y_val)
    val_pred = torch.argmax(val_logits, dim=1)
    val_acc = (val_pred == y_val).float().mean()
    print(f"\nValidation Loss: {val_loss.item():.4f}, Accuracy: {val_acc.item():.4f}")
