import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import random
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load text
with open('nietzsche.txt', 'r') as file:
    text = file.read().lower()
print(f'Text corpus length: {len(text)}')

# Sequence parameters
maxlen = 60
step = 3

# Extract sequences
sentences = []
next_chars = []

for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])

print(f'Number of sequences: {len(sentences)}')

# Character mapping
chars = sorted(list(set(text)))
print(f'Unique characters: {len(chars)}')
char_indices = {char: idx for idx, char in enumerate(chars)}
indices_char = {idx: char for char, idx in char_indices.items()}

# One-hot encode
print('Vectorizing...')
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.float32)
y = np.zeros((len(sentences), len(chars)), dtype=np.float32)

for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

print(f'X shape: {x.shape}, Y shape: {y.shape}')

# PyTorch Model
class LSTMTextGenerator(nn.Module):
    def __init__(self, input_size, hidden_size, num_chars):
        super(LSTMTextGenerator, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.dense = nn.Linear(hidden_size, num_chars)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # Take only the last timestep
        output = self.dense(lstm_out[:, -1, :])
        return output

model = LSTMTextGenerator(len(chars), 128, len(chars)).to(device)
optimizer = optim.RMSprop(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# Convert data to tensors
x_tensor = torch.FloatTensor(x).to(device)
y_tensor = torch.FloatTensor(y).to(device)

# Sampling function
def sample(preds, temperature=1.0):
    """Sample next character based on predictions"""
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# Training loop
batch_size = 128
num_epochs = 60

for epoch in range(1, num_epochs + 1):
    print(f'Epoch {epoch}')
    
    # Training
    model.train()
    total_loss = 0
    num_batches = 0
    
    for i in range(0, len(x_tensor), batch_size):
        batch_x = x_tensor[i:i + batch_size]
        batch_y = y_tensor[i:i + batch_size]
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    print(f'Average loss: {avg_loss:.4f}')
    
    # Text generation
    model.eval()
    start_index = random.randint(0, len(text) - maxlen - 1)
    generated_text = text[start_index: start_index + maxlen]
    print(f'--- Generating with seed: "{generated_text}"')
    
    with torch.no_grad():
        for temperature in [0.2, 0.5, 1.0, 1.2]:
            print(f'------ temperature: {temperature}')
            current_text = generated_text
            sys.stdout.write(current_text)
            
            # Generate 400 characters
            for _ in range(400):
                sampled = np.zeros((1, maxlen, len(chars)), dtype=np.float32)
                for t, char in enumerate(current_text):
                    sampled[0, t, char_indices[char]] = 1.0
                
                sampled_tensor = torch.FloatTensor(sampled).to(device)
                preds = model(sampled_tensor)[0].cpu().numpy()
                preds = np.exp(preds) / np.sum(np.exp(preds))  # Softmax
                next_index = sample(preds, temperature)
                next_char = indices_char[next_index]
                
                current_text += next_char
                current_text = current_text[1:]
                
                sys.stdout.write(next_char)
                sys.stdout.flush()
            
            print()
