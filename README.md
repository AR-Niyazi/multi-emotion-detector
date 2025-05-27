# Multi-Label Emotion Detector (SemEval 2025 Track A)

This project (Multi-Emotion-Detector) aims to classify the presence of 6 basic emotions from text snippets using a multi-label classification approach with BERT.

## Emotions:
- Joy
- Sadness
- Fear
- Anger
- Surprise
- Disgust

## Folder Structure
- `data/`: Raw and processed CSVs
- `src/`: Core source code
- `outputs/`: Saved models, plots, logs
- `notebooks/`: EDA and analysis

## How to Run
- `train.py`: Trains the BERT model
- `evaluate.py`: Evaluates using F1, Hamming loss
- `predict.py`: Generates predictions on test data
