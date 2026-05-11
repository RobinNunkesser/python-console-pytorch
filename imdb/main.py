import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import matplotlib.pyplot as plt
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

np.random.seed(1337)
torch.manual_seed(1337)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load IMDB dataset using torchtext or simple torch datasets
from torchvision import datasets
import pickle

# Alternative: Use Keras-like API for quick loading
try:
    # Try to load using torchtext
    train_iter = IMDB(split='train')
    test_iter = IMDB(split='test')
except:
    # Fallback: Create synthetic data similar to IMDB
    print("Using PyTorch-native IMDB loading...")
    from torch.utils.data import TensorDataset
    
    # For simplicity, we'll load a simplified version
    # In practice, you'd use torchtext or load the actual IMDB dataset
    max_features = 10000
    maxlen = 500
    
    # Loading IMDB from Keras datasets as reference
    import torchvision.datasets as dsets
    # This is a simplified example - in production use torchtext
    
    # Create dummy data for demonstration
    x_train = torch.randint(0, max_features, (20000, maxlen))
    y_train = torch.randint(0, 2, (20000,)).float()
    x_test = torch.randint(0, max_features, (5000, maxlen))
    y_test = torch.randint(0, 2, (5000,)).float()

# For proper IMDB loading, use:
try:
    from torchtext.datasets import IMDB as IMDBDataset
    from torchtext.data.utils import get_tokenizer
    from torchtext.vocab import build_vocab_from_iterator
    
    tokenizer = get_tokenizer("basic_english")
    train_iter = IMDBDataset(split='train')
    
    def yield_tokens(data_iter):
        for _, text in data_iter:
            yield tokenizer(text)
    
    vocab = build_vocab_from_iterator(yield_tokens(IMDBDataset(split='train')), 
                                       specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])
    
    max_features = len(vocab)
    maxlen = 500
    
    def encode_text(text):
        tokens = tokenizer(text)[:maxlen]
        indices = [vocab[token] for token in tokens]
        indices += [0] * (maxlen - len(indices))
        return indices[:maxlen]
    
    # Load data
    x_train_list, y_train_list = [], []
    x_test_list, y_test_list = [], []
    
    for label, text in IMDBDataset(split='train'):
        x_train_list.append(encode_text(text))
        y_train_list.append(float(label - 1))  # Convert 1,2 to 0,1
    
    for label, text in IMDBDataset(split='test'):
        x_test_list.append(encode_text(text))
        y_test_list.append(float(label - 1))
    
    x_train = torch.tensor(x_train_list)
    y_train = torch.tensor(y_train_list)
    x_test = torch.tensor(x_test_list)
    y_test = torch.tensor(y_test_list)
    
except Exception as e:
    print(f"Warning: Could not load IMDB dataset: {e}")
    print("Using simulated data instead...")
    max_features = 10000
    maxlen = 500
    x_train = torch.randint(0, max_features, (20000, maxlen))
    y_train = torch.randint(0, 2, (20000,)).float()
    x_test = torch.randint(0, max_features, (5000, maxlen))
    y_test = torch.randint(0, 2, (5000,)).float()

print(f"x_train shape: {x_train.shape}")
print(f"y_train shape: {y_train.shape}")

# Model
class IMDBNet(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super(IMDBNet, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.embedding(x)
        _, h = self.rnn(x)
        x = h[-1]
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

model = IMDBNet(vocab_size=max_features, embedding_dim=32, hidden_size=32).to(device)
optimizer = optim.RMSprop(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

# Training
x_train = x_train.to(device)
y_train = y_train.to(device)
x_test = x_test.to(device)
y_test = y_test.to(device)

# Use 20% for validation
split_idx = int(len(x_train) * 0.8)
x_train_split = x_train[:split_idx]
y_train_split = y_train[:split_idx]
x_val = x_train[split_idx:]
y_val = y_train[split_idx:]

batch_size = 128
epochs = 10

train_accs = []
val_accs = []
train_losses = []
val_losses = []

for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for i in range(0, len(x_train_split), batch_size):
        batch_x = x_train_split[i:i+batch_size]
        batch_y = y_train_split[i:i+batch_size].unsqueeze(1)
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = (outputs > 0.5).float().squeeze()
        correct += (pred == batch_y.squeeze()).sum().item()
        total += batch_y.size(0)
    
    train_loss = total_loss / (len(x_train_split) // batch_size)
    train_acc = correct / total
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(x_val)
        val_loss = criterion(val_outputs, y_val.unsqueeze(1))
        val_pred = (val_outputs > 0.5).float().squeeze()
        val_correct = (val_pred == y_val).sum().item()
        val_acc = val_correct / len(y_val)
    
    val_losses.append(val_loss.item())
    val_accs.append(val_acc)
    
    print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
          f"Val Loss: {val_loss.item():.4f}, Val Acc: {val_acc:.4f}")

# Plotting
plt.style.use('dark_background')

epochs_range = range(len(train_accs))

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_accs, label="Training")
plt.plot(epochs_range, val_accs, label="Validierung")
plt.title("Korrektklassifizierungsrate Training/Validierung")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend()
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_losses, label="Verlust Training")
plt.plot(epochs_range, val_losses, label="Verlust Validierung")
plt.title("Wert der Verlustfunktion Training/Validierung")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig("train_validation_acc_imdb.svg")
plt.savefig("train_validation_loss_imdb.svg")
