import torch
from torch import nn
from torch import optim
from torchviz import make_dot
from torchview import draw_graph

# Check if an accelerator is available and set the device accordingly
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

# Training data for AND
x_train = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32, device=device)
y_train = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32, device=device).view(-1,1)

print(f"Shape of X: {x_train.shape}")
print(f"Shape of y: {y_train.shape}")

# Define the neural network architecture
class XORNet(nn.Module):
    def __init__(self):
        super(XORNet, self).__init__()
        self.sequential = nn.Sequential(nn.Linear(2, 2),
                      nn.Sigmoid(),
                      nn.Linear(2, 1),
                      nn.Sigmoid())

    def forward(self, x):
        x = self.sequential(x)
        return x



# Initialize the network, loss function, and optimizer
model = XORNet().to(device)
print(model)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Training loop
for epoch in range(10000):
    optimizer.zero_grad()
    y_pred = model(x_train)
    loss = loss_fn(y_pred, y_train)
    loss.backward()
    optimizer.step()

    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# Visualize the model
model_graph = draw_graph(XORNet(), x_train)
dot = model_graph.visual_graph
dot.format = "svg"
dot.render("pytorch_xor_model", cleanup=True)


# Testing the model
with torch.no_grad():
    test_output = model(x_train)
    print(test_output)

# Save the model as ONNX
torch.onnx.export(model, x_train, "xor_model.onnx", input_names=["input"], output_names=["output"])


