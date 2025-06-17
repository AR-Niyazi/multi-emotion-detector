#!/usr/bin/env python3
"""
Test script for data utilities
Run this script to verify all data processing functions work correctly
"""

import sys
import os

sys.path.append('src')

from data_utils import (
    load_dataset, preprocess_texts, get_emotion_statistics,
    split_dataset, load_tokenizer, get_dataloader, preprocess_pipeline
)


def test_data_loading():
    """Test data loading functionality"""
    print("Testing data loading...")

    try:
        df = load_dataset('data/raw/track-a.csv')
        print(f"✓ Successfully loaded {len(df)} samples")

        # Check required columns
        required_cols = ['id', 'text', 'anger', 'fear', 'joy', 'sadness', 'surprise']
        for col in required_cols:
            assert col in df.columns, f"Missing column: {col}"
        print("✓ All required columns present")

        return df
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return None


def test_text_preprocessing(df):
    """Test text preprocessing"""
    print("\nTesting text preprocessing...")

    try:
        df_processed = preprocess_texts(df)
        print(f"✓ Text preprocessing completed")
        print(f"✓ Dataset size after cleaning: {len(df_processed)}")

        # Check that texts are cleaned
        sample_text = df_processed['text'].iloc[0]
        print(f"✓ Sample processed text: '{sample_text[:50]}...'")

        return df_processed
    except Exception as e:
        print(f"✗ Error in text preprocessing: {e}")
        return None


def test_emotion_statistics(df):
    """Test emotion statistics calculation"""
    print("\nTesting emotion statistics...")

    try:
        stats = get_emotion_statistics(df)
        print("✓ Emotion statistics calculated")

        # Display some stats
        for emotion in ['anger', 'fear', 'joy', 'sadness', 'surprise']:
            count = stats[emotion]['count']
            pct = stats[emotion]['percentage']
            print(f"  {emotion}: {count} samples ({pct:.1f}%)")

        return stats
    except Exception as e:
        print(f"✗ Error calculating statistics: {e}")
        return None


def test_data_splitting(df):
    """Test dataset splitting"""
    print("\nTesting dataset splitting...")

    try:
        train_df, val_df, test_df = split_dataset(df)

        total_samples = len(train_df) + len(val_df) + len(test_df)
        assert total_samples == len(df), "Sample count mismatch after splitting"

        print(f"✓ Dataset split successfully:")
        print(f"  Train: {len(train_df)} samples")
        print(f"  Validation: {len(val_df)} samples")
        print(f"  Test: {len(test_df)} samples")

        return train_df, val_df, test_df
    except Exception as e:
        print(f"✗ Error splitting dataset: {e}")
        return None, None, None


def test_tokenizer_and_dataloader(train_df):
    """Test tokenizer loading and dataloader creation"""
    print("\nTesting tokenizer and dataloader...")

    try:
        # Load tokenizer
        tokenizer = load_tokenizer()
        print("✓ Tokenizer loaded successfully")

        # Create a small sample for testing
        sample_df = train_df.head(32)  # Small batch for testing

        # Create dataloader
        dataloader = get_dataloader(sample_df, tokenizer, batch_size=8, shuffle=False)
        print("✓ DataLoader created successfully")

        # Test one batch
        batch = next(iter(dataloader))
        print(f"✓ Batch loaded successfully:")
        print(f"  Input IDs shape: {batch['input_ids'].shape}")
        print(f"  Attention mask shape: {batch['attention_mask'].shape}")
        print(f"  Labels shape: {batch['labels'].shape}")

        return True
    except Exception as e:
        print(f"✗ Error with tokenizer/dataloader: {e}")
        return False


def test_full_pipeline():
    """Test the complete preprocessing pipeline"""
    print("\nTesting complete preprocessing pipeline...")

    try:
        # Run the full pipeline
        summary = preprocess_pipeline('data/raw/track-a.csv')

        print("✓ Full pipeline completed successfully!")
        print(f"✓ Processed {summary['processed_size']} samples")
        print(f"✓ Split into train({summary['train_size']}), val({summary['val_size']}), test({summary['test_size']})")

        # Check if processed files exist
        for split in ['train', 'val', 'test']:
            filepath = f"data/processed/{split}.csv"
            if os.path.exists(filepath):
                print(f"✓ {split}.csv saved successfully")
            else:
                print(f"✗ {split}.csv not found")

        return True
    except Exception as e:
        print(f"✗ Error in full pipeline: {e}")
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("EMOTION DETECTION - DATA UTILITIES TEST")
    print("=" * 60)

    # Test 1: Data loading
    df = test_data_loading()
    if df is None:
        print("Cannot proceed without data. Please check your CSV file path.")
        return

    # Test 2: Text preprocessing
    df_processed = test_text_preprocessing(df)
    if df_processed is None:
        print("Text preprocessing failed.")
        return

    # Test 3: Emotion statistics
    stats = test_emotion_statistics(df_processed)
    if stats is None:
        print("Statistics calculation failed.")
        return

    # Test 4: Data splitting
    train_df, val_df, test_df = test_data_splitting(df_processed)
    if train_df is None:
        print("Data splitting failed.")
        return

    # Test 5: Tokenizer and DataLoader
    dataloader_success = test_tokenizer_and_dataloader(train_df)
    if not dataloader_success:
        print("Tokenizer/DataLoader test failed.")
        return

    # Test 6: Full pipeline
    pipeline_success = test_full_pipeline()

    print("\n" + "=" * 60)
    if pipeline_success:
        print("🎉 ALL TESTS PASSED! Data utilities are working correctly.")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    print("=" * 60)


if __name__ == "__main__":
    main()