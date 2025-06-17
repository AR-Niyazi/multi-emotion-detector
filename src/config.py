# src/config.py

# Data Config
DATA_CONFIG = {
    'csv_path': 'data/raw/track-a.csv',
    'text_column': 'text',
    'emotion_columns': ['joy', 'sadness', 'fear', 'anger', 'surprise'],
    'test_size': 0.2,
    'validation_size': 0.1,
    'random_state': 42
}

# Preprocessing configuration
PREPROCESSING_CONFIG = {
    'remove_stopwords': True,
    'apply_lemmatization': True,
    'lowercase': True,
    'remove_punctuation': True,
    'remove_urls': True,
    'min_word_length': 2
}

# Output paths
OUTPUT_PATHS = {
    'processed_data': 'outputs/processed_data.pkl',
    'model_path': 'outputs/bert_model/',
    'results_path': 'outputs/results.json'
}

# DATA_PATH = "../data/raw/track-a.csv"
# PROCESSED_PATH = "data/processed/"
# MODEL_SAVE_PATH = "outputs/models/bert_emotion.pt"
# PLOTS_PATH = "outputs/plots/"
# MAX_LEN = 128
# BATCH_SIZE = 16
# EPOCHS = 3
# LR = 2e-5
# NUM_LABELS = 5 # because of 5 emotion labels in the dataset
# SEED = 42
