# transfer_learning_cifar10

Transfer Learning Demo mit PyTorch:
- Backbone: `ResNet18` (ImageNet-pretrained)
- Datensatz: `CIFAR-10`
- Zweiphasiges Training:
  1. nur Klassifikationskopf (`fc`) trainieren
  2. letzten Residual-Block (`layer4`) + Kopf feinjustieren

## Installation

```bash
pip install torch torchvision
```

## Start

```bash
python main.py --epochs-head 2 --epochs-finetune 2
```

## Didaktischer Fokus

- Warum Transfer Learning in der Praxis oft besser als Training from scratch ist
- Unterschied zwischen Feature-Extraktion und Fine-Tuning
- Lernratenstrategie für unterschiedliche Phasen
