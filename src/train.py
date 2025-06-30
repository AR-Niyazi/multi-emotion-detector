from transformers import BertTokenizer
from model import get_model
from config import MODEL_PATH
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import Trainer, TrainingArguments
import inspect
print("⚠️ TrainingArguments loaded from:", inspect.getfile(TrainingArguments))
from torch.utils.data import Dataset
import numpy as np
import os

# Base directory of the project
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
TRAIN_PATH = os.path.join(BASE_DIR, "data/processed/train.csv")
VAL_PATH = os.path.join(BASE_DIR, "data/processed/val.csv")


# List of emotion columns
LABEL_COLUMNS = ["anger", "fear", "joy", "sadness", "surprise"]

class EmotionDataset(Dataset):
    """
    Custom PyTorch dataset for loading emotion-labeled text samples.
    """
    def __init__(self, dataframe, tokenizer, max_length=128):
        self.texts = dataframe["text"].tolist()
        self.labels = dataframe[LABEL_COLUMNS].values
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        labels = self.labels[idx]
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.FloatTensor(labels)
        }

def compute_metrics(pred):
    from sklearn.metrics import f1_score, hamming_loss

    logits, labels = pred
    preds = (torch.sigmoid(torch.tensor(logits)) > 0.5).int()
    labels = torch.tensor(labels)

    return {
        "f1_micro": f1_score(labels, preds, average="micro"),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "hamming_loss": hamming_loss(labels, preds)
    }

def train_model():
    # Load preprocessed data
    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)

    # Load tokenizer and model
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = get_model(num_labels=5)


    # Create datasets
    train_dataset = EmotionDataset(train_df, tokenizer)
    val_dataset = EmotionDataset(val_df, tokenizer)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./outputs/checkpoints",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        learning_rate=2e-5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1_micro",
        save_total_limit=2,
        report_to="none"  # Disable W&B, etc.
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # Train the model
    trainer.train()

    # Save final model
    #model = get_model(num_labels=len(LABEL_COLUMNS))
    model.save_pretrained(MODEL_PATH)
    tokenizer.save_pretrained(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_model()
