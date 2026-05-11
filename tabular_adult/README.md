# tabular_adult

PyTorch-Beispiel für Tabular Deep Learning mit kategorialen Embeddings:
- Datensatz: UCI Adult (`fetch_openml("adult")`)
- Numerische Features: Standardisierung
- Kategoriale Features: Embedding pro Spalte
- Modell: MLP auf konkatenierten numerischen + eingebetteten Features

## Installation

```bash
pip install torch pandas scikit-learn
```

## Start

```bash
python main.py --epochs 10 --batch-size 256
```

## Didaktischer Fokus

- Warum Embeddings für kategoriale Tabellenmerkmale nützlich sind
- Trennung von numerischem und kategorischem Featurepfad
- Vergleich mit klassischen Baselines (z. B. Logistic Regression, XGBoost) als Übungsaufgabe
