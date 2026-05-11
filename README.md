# python-console-pytorch

Diese Sammlung enthält Implementierungen von Deep Learning und Machine Learning Konzepten mit PyTorch für die Vorlesung Künstliche Intelligenz.

## Struktur

### Logische Gatter (Grundlagen)
- **and/**: AND-Gatter mit linearem Netzwerk
- **not/**: NOT-Gatter
- **or/**: OR-Gatter mit linearem Netzwerk
- **xor/**: XOR-Gatter mit mehrlayerigem Netzwerk

### Deep Learning Beispiele
- **cartpole/**: Actor-Critic Reinforcement Learning mit CartPole-Umgebung
- **mnist/**: Convolutional Neural Network für Bildklassifikation
- **imdb/**: SimpleRNN für Sentimentanalyse von IMDB-Reviews
- **nietzsche/**: LSTM für Textgenerierung basierend auf Nietzsche-Texten

### Spezialisierte Trainingsverfahren
- **custom_training/**: Manueller Training-Loop mit GradientTape (torch.autograd)

## Installation

```bash
pip install torch torchvision torchaudio numpy matplotlib torchtext
```

## Quick Start

### 1. Logische Gatter

```bash
# AND-Gatter (linear separierbar)
cd and && python main.py

# XOR-Gatter (nicht-linear, braucht hidden layer)
cd ../xor && python main.py
```

Output: Contour-Plots der Netzwerk-Entscheidungsgrenzen

### 2. MNIST Bildklassifikation

```bash
cd mnist && python main.py
```

- Downloadet MNIST-Dataset automatisch
- CNN mit 2 Convolutional Layern
- Training: ~10 Epochen, ~95% Accuracy erreichbar

### 3. CartPole Reinforcement Learning

```bash
cd cartpole && python actorcritic.py
```

- Agent lernt, eine Stange zu balancieren
- Actor-Critic Architektur
- Visualisiert Trainingsfortschritt pro Episode

### 4. Text Generation (Nietzsche)

```bash
cd nietzsche && python main.py
```

- LSTM trainiert auf Nietzsche-Texten
- Generiert Text mit verschiedenen Temperaturen
- 60 Epochen Training

### 5. Manuelles Training (Custom Training Loop)

```bash
cd custom_training && python main.py
```

- Zeigt expliziten Training-Loop statt `model.fit()`
- Perfekt um PyTorch's Autograd zu verstehen
- `optimizer.zero_grad()` → `loss.backward()` → `optimizer.step()`

## PyTorch Grundkonzepte

### 1. Modelle definieren

```python
class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

model = MyNet().to(device)
```

### 2. Standard Training-Loop

```python
for epoch in range(epochs):
    model.train()
    for batch_x, batch_y in train_loader:
        # 1. Forward pass
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        
        # 2. Backward pass (Autograd berechnet Gradienten)
        loss.backward()
        
        # 3. Update parameters
        optimizer.step()
```

### 3. Evaluation

```python
model.eval()
with torch.no_grad():  # Keine Gradienten für Inferenz
    predictions = model(test_data)
```

## Unterschiede zu Keras/TensorFlow

| Aspekt | PyTorch | Keras |
|--------|---------|-------|
| Style | Imperativ, pythonic | Deklarativ, sequentiell |
| Training | Expliziter Loop | `model.fit()` |
| Gradienten | `backward()` + Autograd | `GradientTape` |
| Debugging | Einfach (Standard Python) | Komplexer (Graph-Modus) |
| Default | GPU-Support | CPU-Default |
| Community | Forschung > Production | Education > Research |

## Tipps & Tricks

1. **GPU nutzen**: Alle Tensoren müssen auf gleichem Device sein
   ```python
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   tensor.to(device)
   ```

2. **Gradienten löschen**: Immer vor `backward()` aufrufen
   ```python
   optimizer.zero_grad()  # Wichtig!
   ```

3. **Evaluation Mode**: Mit `torch.no_grad()` für schnellere Inferenz
   ```python
   model.eval()
   with torch.no_grad():
       # Keine Gradienten-Berechnung
   ```

4. **Learning Rate Scheduling**: Optimizer-Parameter anpassen
   ```python
   scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
   ```

## Ressourcen

- [PyTorch Official Documentation](https://pytorch.org/docs/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Fast.ai Practical Deep Learning Course](https://course.fast.ai)

## Lizenz

Diese Beispiele sind Teil der Vorlesung "Künstliche Intelligenz" (Prof. Dr. Robin Nunkesser, 2026).