# src/data_utils.py

import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from config import DATA_PATH, MAX_LEN

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

def tokenize_texts(texts):
    return tokenizer(texts, padding='max_length', truncation=True, max_length=MAX_LEN, return_tensors="pt")
