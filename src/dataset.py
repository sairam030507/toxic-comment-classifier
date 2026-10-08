"""
Data ingestion, cleaning, stratification, and dataset preparation module.
Supports:
1. Primary Kaggle Jigsaw Toxic Comment Classification Challenge dataset (6 labels)
2. Extra Jigsaw Unintended Bias dataset (for fairness, bias auditing, and data augmentation)
"""

import os
import re
import json
import urllib.request
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional
from sklearn.model_selection import train_test_split

LABEL_COLUMNS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

DEMOGRAPHIC_TERMS = [
    'gay', 'lesbian', 'homosexual', 'queer', 'lgbt', 'lgbtq',
    'muslim', 'jewish', 'islam', 'jew', 'christian', 'hindu', 'buddhist',
    'black', 'white', 'african', 'african-american', 'asian', 'latino', 'hispanic',
    'woman', 'female', 'girl', 'man', 'male', 'boy', 'transgender', 'trans',
    'feminist', 'feminism', 'disabled', 'disability', 'immigrant', 'refugee'
]

PRIMARY_DATA_URL = "https://huggingface.co/datasets/tasksource/jigsaw_toxicity/resolve/main/train.csv"
EXTRA_DATA_URL = "https://huggingface.co/datasets/james-burton/jigsaw_unintended_bias100K/resolve/main/data/train-00000-of-00001-18ff1ad7d1d0c2c1.parquet"


def clean_text_minimal(text: str) -> str:
    """
    Cleans text minimally according to Section 5.1 Step 4 of the proposal:
    - Normalizes whitespace
    - Strips wiki markup artefacts
    - Deliberately preserves tokens, punctuation, and casing needed for explainability
    """
    if not isinstance(text, str):
        return ""
    
    # Replace URLs
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    # Remove IP addresses
    text = re.sub(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', ' ', text)
    # Strip wiki section headers (e.g., == Section ==, === Sub ===)
    text = re.sub(r'={2,}\s*.*?\s*={2,}', ' ', text)
    # Strip wiki links format: [[Target|Label]] -> Label, [[Target]] -> Target
    text = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]+)\]\]', r'\1', text)
    # Strip wiki templates {{...}}
    text = re.sub(r'\{\{.*?\}\}', ' ', text)
    # Strip HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Normalize multiple whitespace, tabs, and newlines to a single space
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def download_file(url: str, dest_path: str):
    """Downloads a file if it does not already exist."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1000:
        print(f"File already cached: {dest_path}")
        return
    print(f"Downloading {url} to {dest_path}...")
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as resp, open(dest_path, 'wb') as out_f:
        chunk_size = 1024 * 1024
        downloaded = 0
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            out_f.write(chunk)
            downloaded += len(chunk)
            print(f"Downloaded {downloaded / (1024 * 1024):.1f} MB", end='\r')
    print(f"\nDownload complete: {dest_path}")


def load_primary_dataset(raw_dir: str = "data/raw", sample_size: Optional[int] = 30000) -> pd.DataFrame:
    """
    Loads the primary Jigsaw dataset, cleans comments minimally, and optionally
    stratifies/subsamples according to Section 5.1 of the proposal.
    """
    cache_path = os.path.join(raw_dir, "jigsaw_toxic_train.csv")
    download_file(PRIMARY_DATA_URL, cache_path)
    
    print("Loading primary Jigsaw dataset...")
    df = pd.read_csv(cache_path)
    # Retain label columns and drop comment id (Section 5.1 Step 1)
    df = df[['comment_text'] + LABEL_COLUMNS].dropna(subset=['comment_text']).copy()
    
    # Minimal text cleaning
    df['comment_text'] = df['comment_text'].apply(clean_text_minimal)
    df = df[df['comment_text'].str.len() > 5].copy()
    
    # Calculate positive label count for stratified subsampling (Section 5.1 Step 2)
    df['pos_count'] = df[LABEL_COLUMNS].sum(axis=1)
    
    if sample_size and len(df) > sample_size:
        print(f"Subsampling to {sample_size} preserving class distribution...")
        # Oversample rare labels (threat, identity_hate, severe_toxic)
        rare_mask = (df['threat'] == 1) | (df['identity_hate'] == 1) | (df['severe_toxic'] == 1)
        rare_df = df[rare_mask]
        
        # Sample remaining from non-rare to reach sample_size
        remaining_needed = sample_size - len(rare_df)
        other_df = df[~rare_mask]
        
        # Stratify other by pos_count (0 = benign, 1 = single toxic, 2+ = multi-label)
        strat_col = other_df['pos_count'].clip(upper=2)
        other_sample, _ = train_test_split(
            other_df,
            train_size=remaining_needed,
            stratify=strat_col,
            random_state=42
        )
        df = pd.concat([rare_df, other_sample], ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    return df


def load_extra_bias_dataset(raw_dir: str = "data/raw", sample_size: int = 10000) -> pd.DataFrame:
    """
    Loads extra Jigsaw Unintended Bias dataset for identity-term fairness auditing
    and multi-label balance enrichment.
    """
    cache_path = os.path.join(raw_dir, "jigsaw_unintended_bias.parquet")
    download_file(EXTRA_DATA_URL, cache_path)
    
    print("Loading extra Jigsaw Unintended Bias dataset...")
    df = pd.read_parquet(cache_path)
    df = df.dropna(subset=['comment_text']).copy()
    df['comment_text'] = df['comment_text'].apply(clean_text_minimal)
    df = df[df['comment_text'].str.len() > 5].copy()
    
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        
    return df


def prepare_stratified_splits(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits into train / validation / test partitions (Section 5.1 Step 3),
    stratifying so that rare labels appear in every split.
    Uses multi-label composite strata.
    """
    # Create composite stratification key based on label pattern
    # For very rare combinations, cluster by pos_count and rare label flags
    strata = (
        df['threat'].astype(str) + "_" +
        df['identity_hate'].astype(str) + "_" +
        df['severe_toxic'].astype(str) + "_" +
        (df[LABEL_COLUMNS].sum(axis=1) > 0).astype(str)
    )
    
    # Handle rare strata with < 3 samples by grouping into 'other'
    counts = strata.value_counts()
    valid_strata = counts[counts >= 3].index
    strata_mapped = strata.apply(lambda s: s if s in valid_strata else 'other')

    # First split: train+val vs test
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=strata_mapped,
        random_state=random_state
    )

    # Recompute strata for second split
    strata_train_val = (
        train_val_df['threat'].astype(str) + "_" +
        train_val_df['identity_hate'].astype(str) + "_" +
        train_val_df['severe_toxic'].astype(str) + "_" +
        (train_val_df[LABEL_COLUMNS].sum(axis=1) > 0).astype(str)
    )
    counts_tv = strata_train_val.value_counts()
    valid_tv = counts_tv[counts_tv >= 2].index
    strata_tv_mapped = strata_train_val.apply(lambda s: s if s in valid_tv else 'other')

    val_relative_size = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_relative_size,
        stratify=strata_tv_mapped,
        random_state=random_state
    )

    print(f"Dataset split complete: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def calculate_positive_class_weights(df: pd.DataFrame) -> Dict[str, float]:
    """
    Computes positive-class weights inversely proportional to label frequency:
    pos_weight = (N_total - N_pos) / N_pos
    Used in BCEWithLogitsLoss to handle severe class imbalance (Section 5.2).
    """
    weights = {}
    n_total = len(df)
    for col in LABEL_COLUMNS:
        n_pos = df[col].sum()
        n_neg = n_total - n_pos
        weight = float(n_neg / max(n_pos, 1))
        # Cap weight between 1.0 and 50.0 to prevent gradient explosion
        weights[col] = float(np.clip(weight, 1.0, 50.0))
    return weights


def get_identity_terms_mask(texts: pd.Series) -> np.ndarray:
    """Returns boolean mask indicating if each text contains at least one demographic identity term."""
    pattern = r'\b(?:' + '|'.join(DEMOGRAPHIC_TERMS) + r')\b'
    regex = re.compile(pattern, re.IGNORECASE)
    return texts.apply(lambda t: bool(regex.search(str(t)))).to_numpy()
