import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

np.random.seed(1337)
torch.manual_seed(1337)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training data for NOT
x_train = torch.tensor([[0], [1]], dtype=torch.float32, device=device)
y_train = torch.tensor([[1], [0]], dtype=torch.float32, device=device)

# Model - linear for NOT
class NOTNet(nn.Module):
    def __init__(self):
        super(NOTNet, self).__init__()
        self.fc1 = nn.Linear(1, 1)
        
    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        return x

model = NOTNet().to(device)
optimizer = optim.Adam(model.parameters())
criterion = nn.MSELoss()

# Training
epochs = 100
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(x_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

# Print predictions
model.eval()
with torch.no_grad():
    predictions = model(x_train)
    print("Predictions for NOT:")
    print(predictions)
