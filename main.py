from transformers import BertTokenizer, BertForSequenceClassification

from src.config import MODEL_PATH

model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
