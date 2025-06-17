import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from typing import List, Tuple, Dict
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import warnings

warnings.filterwarnings('ignore')


class EmotionDataset(Dataset):
    """Custom Dataset class for emotion detection"""

    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        labels = torch.FloatTensor(self.labels[idx])

        # Tokenize the text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': labels
        }


def load_dataset(csv_path: str) -> pd.DataFrame:
    """
    Load the emotion dataset from CSV file

    Args:
        csv_path (str): Path to the CSV file

    Returns:
        pd.DataFrame: Loaded dataset
    """
    try:
        df = pd.read_csv(csv_path)
        print(f"Dataset loaded successfully! Shape: {df.shape}")

        # Verify required columns exist
        required_cols = ['id', 'text', 'anger', 'fear', 'joy', 'sadness', 'surprise']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        return df

    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file not found at path: {csv_path}")
    except Exception as e:
        raise Exception(f"Error loading dataset: {str(e)}")


def clean_text(text: str) -> str:
    """
    Clean and preprocess individual text

    Args:
        text (str): Raw text to clean

    Returns:
        str: Cleaned text
    """
    if pd.isna(text):
        return ""

    # Convert to string and lowercase
    text = str(text).lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # Remove user mentions and hashtags (keep the content)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)

    # Remove extra whitespace and newlines
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()

    # Remove very short texts (less than 3 characters)
    if len(text) < 3:
        return ""

    return text


def preprocess_texts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply text preprocessing to the entire dataset

    Args:
        df (pd.DataFrame): Dataset with text column

    Returns:
        pd.DataFrame: Dataset with cleaned text
    """
    print("Starting text preprocessing...")

    # Create a copy to avoid modifying original
    df_processed = df.copy()

    # Apply text cleaning
    df_processed['text'] = df_processed['text'].apply(clean_text)

    # Remove rows with empty text after cleaning
    initial_count = len(df_processed)
    df_processed = df_processed[df_processed['text'] != ""]
    final_count = len(df_processed)

    print(f"Preprocessing complete! Removed {initial_count - final_count} empty/invalid texts")
    print(f"Final dataset size: {final_count} samples")

    return df_processed.reset_index(drop=True)


def get_emotion_statistics(df: pd.DataFrame) -> Dict:
    """
    Calculate statistics for emotion labels

    Args:
        df (pd.DataFrame): Dataset with emotion columns

    Returns:
        Dict: Statistics about emotion distribution
    """
    emotion_cols = ['anger', 'fear', 'joy', 'sadness', 'surprise']

    stats = {}

    # Individual emotion counts
    for emotion in emotion_cols:
        stats[emotion] = {
            'count': df[emotion].sum(),
            'percentage': (df[emotion].sum() / len(df)) * 100
        }

    # Multi-label statistics
    df['total_emotions'] = df[emotion_cols].sum(axis=1)
    stats['multi_label'] = {
        'avg_emotions_per_text': df['total_emotions'].mean(),
        'max_emotions_per_text': df['total_emotions'].max(),
        'no_emotion_count': (df['total_emotions'] == 0).sum(),
        'single_emotion_count': (df['total_emotions'] == 1).sum(),
        'multi_emotion_count': (df['total_emotions'] > 1).sum()
    }

    return stats


def create_stratification_key(df: pd.DataFrame) -> pd.Series:
    """
    Create a stratification key for multi-label data

    Args:
        df (pd.DataFrame): Dataset with emotion columns

    Returns:
        pd.Series: Stratification keys
    """
    emotion_cols = ['anger', 'fear', 'joy', 'sadness', 'surprise']

    # Create a string representation of each label combination
    stratify_key = df[emotion_cols].apply(
        lambda row: ''.join(row.astype(str)), axis=1
    )

    # Get value counts to identify rare combinations
    key_counts = stratify_key.value_counts()

    # For combinations that appear less than 2 times, group them as 'rare'
    # This prevents stratification errors
    rare_keys = key_counts[key_counts < 2].index
    stratify_key = stratify_key.replace(rare_keys, 'rare_combination')

    print(f"Created {len(stratify_key.unique())} stratification groups")
    print(f"Rare combinations grouped: {len(rare_keys)}")

    return stratify_key


def split_dataset(df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.1,
                  random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into train, validation, and test sets with robust stratification

    Args:
        df (pd.DataFrame): Complete dataset
        test_size (float): Proportion for test set
        val_size (float): Proportion for validation set (from remaining data)
        random_state (int): Random seed for reproducibility

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: train, val, test dataframes
    """
    print(f"Splitting dataset: Train={1 - test_size - val_size:.1%}, Val={val_size:.1%}, Test={test_size:.1%}")

    try:
        # Create stratification key
        stratify_key = create_stratification_key(df)

        # First split: separate test set
        train_val_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_key
        )

        # Create stratification key for remaining data
        train_val_stratify = create_stratification_key(train_val_df)

        # Second split: separate validation from training
        val_ratio = val_size / (1 - test_size)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_ratio,
            random_state=random_state,
            stratify=train_val_stratify
        )

    except ValueError as e:
        print(f"Stratified split failed: {e}")
        print("Falling back to random split...")

        # Fallback to random split without stratification
        train_val_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state
        )

        val_ratio = val_size / (1 - test_size)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_ratio,
            random_state=random_state
        )

    print(f"Split complete!")
    print(f"Train: {len(train_df)} samples")
    print(f"Validation: {len(val_df)} samples")
    print(f"Test: {len(test_df)} samples")

    return train_df, val_df, test_df


def get_dataloader(df: pd.DataFrame, tokenizer, batch_size: int = 16,
                   max_length: int = 128, shuffle: bool = True) -> DataLoader:
    """
    Create PyTorch DataLoader for the dataset

    Args:
        df (pd.DataFrame): Dataset
        tokenizer: BERT tokenizer
        batch_size (int): Batch size for training
        max_length (int): Maximum sequence length
        shuffle (bool): Whether to shuffle the data

    Returns:
        DataLoader: PyTorch DataLoader object
    """
    emotion_cols = ['anger', 'fear', 'joy', 'sadness', 'surprise']

    texts = df['text'].tolist()
    labels = df[emotion_cols].values.astype(float)

    dataset = EmotionDataset(texts, labels, tokenizer, max_length)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0  # Set to 0 to avoid multiprocessing issues
    )

    print(f"DataLoader created with {len(dataset)} samples, batch_size={batch_size}")

    return dataloader


def save_processed_data(train_df: pd.DataFrame, val_df: pd.DataFrame,
                        test_df: pd.DataFrame, output_dir: str = "data/processed/"):
    """
    Save processed datasets to CSV files

    Args:
        train_df, val_df, test_df: Processed datasets
        output_dir (str): Directory to save processed files
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    train_df.to_csv(f"{output_dir}/train.csv", index=False)
    val_df.to_csv(f"{output_dir}/val.csv", index=False)
    test_df.to_csv(f"{output_dir}/test.csv", index=False)

    print(f"Processed datasets saved to {output_dir}")


def load_tokenizer(model_name: str = "bert-base-uncased"):
    """
    Load BERT tokenizer

    Args:
        model_name (str): Name of the pre-trained model

    Returns:
        tokenizer: BERT tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"Tokenizer loaded: {model_name}")
    return tokenizer


# Main preprocessing pipeline function
def preprocess_pipeline(csv_path: str, output_dir: str = "data/processed/") -> Dict:
    """
    Complete preprocessing pipeline

    Args:
        csv_path (str): Path to raw CSV file
        output_dir (str): Directory to save processed data

    Returns:
        Dict: Summary of preprocessing results
    """
    print("=" * 50)
    print("EMOTION DETECTION - DATA PREPROCESSING PIPELINE")
    print("=" * 50)

    # Step 1: Load dataset
    df = load_dataset(csv_path)

    # Step 2: Preprocess texts
    df_clean = preprocess_texts(df)

    # Step 3: Get statistics
    stats = get_emotion_statistics(df_clean)

    # Step 4: Split dataset
    train_df, val_df, test_df = split_dataset(df_clean)

    # Step 5: Save processed data
    save_processed_data(train_df, val_df, test_df, output_dir)

    # Return summary
    summary = {
        'original_size': len(df),
        'processed_size': len(df_clean),
        'train_size': len(train_df),
        'val_size': len(val_df),
        'test_size': len(test_df),
        'emotion_stats': stats
    }

    print("\nPreprocessing pipeline completed successfully!")
    return summary