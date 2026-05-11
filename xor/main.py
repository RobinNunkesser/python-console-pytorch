import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

np.random.seed(1337)
torch.manual_seed(1337)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training data for XOR
x_train = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32, device=device)
y_train = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32, device=device)

# Model
class XORNet(nn.Module):
    def __init__(self):
        super(XORNet, self).__init__()
        self.fc1 = nn.Linear(2, 2)
        self.fc2 = nn.Linear(2, 1)
        
    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x

model = XORNet().to(device)
optimizer = optim.Adam(model.parameters())
criterion = nn.MSELoss()

# Training
epochs = 10000
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
    print("Predictions for XOR:")
    print(predictions)

# Visualization
model.eval()
x = np.linspace(-0.01, 1.01, 103)
X1_raster, X2_raster = np.meshgrid(x, x)
X1_vektor = X1_raster.flatten()
X2_vektor = X2_raster.flatten()

eingangswerte_grafik = np.vstack((X1_vektor, X2_vektor)).T
eingangswerte_tensor = torch.from_numpy(eingangswerte_grafik).float().to(device)

with torch.no_grad():
    ausgangswerte_grafik = model(eingangswerte_tensor).cpu().numpy().reshape(X1_raster.shape)

# Set dummy limits
ausgangswerte_grafik[0, 0] = 1.25
ausgangswerte_grafik[102, 102] = 0.0

plt.style.use('dark_background')
plt.contourf(X1_raster, X2_raster, ausgangswerte_grafik, 100, cmap="jet")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel("Eingabewert $x_1$")
plt.ylabel("Eingabewert $x_2$")
plt.colorbar()
plt.tight_layout()
plt.savefig("predictions_xor_10000.svg")


