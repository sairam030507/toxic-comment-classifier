"""
End-to-End Training and Pipeline Orchestrator.
1. Ingests primary Jigsaw + extra Unintended Bias dataset.
2. Performs minimal cleaning and stratified splitting (Train / Val / Test).
3. Calculates positive-class weights (pos_weight) for BCE loss.
4. Trains TF-IDF + Logistic Multi-Label Baseline.
5. Trains / Fine-tunes DistilBERT with BCEWithLogitsLoss + pos_weight.
6. Executes independent Threshold Tuning on validation set.
7. Executes Identity-Term Fairness & Bias Audit.
8. Generates 20 LIME/Token Attribution explanations.
"""

import os
import sys
import json
import time
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Any
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertConfig, get_linear_schedule_with_warmup

# Ensure current dir is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset import (
    load_primary_dataset,
    load_extra_bias_dataset,
    prepare_stratified_splits,
    calculate_positive_class_weights,
    LABEL_COLUMNS
)
from src.model import (
    TfidfLogisticMultiLabelClassifier,
    DistilBertForMultiLabelClassification
)
from src.threshold_tuner import (
    tune_thresholds_for_labels,
    compare_flat_vs_tuned,
    save_thresholds
)
from src.fairness_bias import (
    audit_identity_bias_on_test_set,
    run_counterfactual_fairness_probes
)
from src.evaluation import evaluate_predictions
from src.explainability import TokenAttributionExplainer


class ToxicCommentsTorchDataset(Dataset):
    def __init__(self, texts: List[str], labels: np.ndarray, tokenizer, max_len: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding='max_length',
            return_tensors='pt'
        )
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
        }
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item


def train_distilbert_model(
    train_texts: List[str],
    train_labels: np.ndarray,
    val_texts: List[str],
    val_labels: np.ndarray,
    pos_weights: Dict[str, float],
    output_dir: str = "models/model_artifacts/distilbert_multilabel",
    epochs: int = 2,
    batch_size: int = 32,
    max_len: int = 128,
    lr: float = 3e-5
) -> DistilBertForMultiLabelClassification:
    """Trains DistilBERT with positive class weights and mixed precision."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[DistilBERT] Training on device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    config = DistilBertConfig.from_pretrained('distilbert-base-uncased')
    config.num_labels = 6

    # Convert positive weights to tensor
    pos_weight_tensor = torch.tensor(
        [pos_weights[col] for col in LABEL_COLUMNS],
        dtype=torch.float,
        device=device
    )

    model = DistilBertForMultiLabelClassification.from_pretrained(
        'distilbert-base-uncased',
        config=config,
        pos_weight=pos_weight_tensor
    ).to(device)

    train_ds = ToxicCommentsTorchDataset(train_texts, train_labels, tokenizer, max_len=max_len)
    val_ds = ToxicCommentsTorchDataset(val_texts, val_labels, tokenizer, max_len=max_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)

    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None

    print(f"[DistilBERT] Starting training for {epochs} epochs ({total_steps} steps)...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        start_time = time.time()
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            if scaler:
                with torch.amp.autocast('cuda'):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs['loss']
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs['loss']
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            scheduler.step()
            total_loss += loss.item()

            if (step + 1) % 50 == 0 or (step + 1) == len(train_loader):
                elapsed = time.time() - start_time
                print(f"Epoch {epoch+1}/{epochs} | Step {step+1}/{len(train_loader)} | Batch Loss: {loss.item():.4f} | Elapsed: {elapsed:.1f}s", end='\r')

        avg_train_loss = total_loss / len(train_loader)
        print(f"\nEpoch {epoch+1}/{epochs} Complete. Avg Train Loss: {avg_train_loss:.4f}")

    # Save fine-tuned model and tokenizer
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[DistilBERT] Saved model and tokenizer to {output_dir}")
    return model


def predict_distilbert(model, tokenizer, texts: List[str], batch_size: int = 64, max_len: int = 128) -> np.ndarray:
    """Predicts probabilities for a list of texts using DistilBERT."""
    device = next(model.parameters()).device
    model.eval()
    all_probs = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        enc = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=max_len,
            return_tensors='pt'
        )
        input_ids = enc['input_ids'].to(device)
        attention_mask = enc['attention_mask'].to(device)
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = outputs['probabilities'].cpu().numpy()
            all_probs.append(probs)
            
    return np.vstack(all_probs)


def main():
    print("=" * 70)
    print("MULTI-LABEL TOXIC COMMENT CLASSIFICATION PIPELINE")
    print("Explainability & Fairness Analysis | Proposal Checkpoints 1-4")
    print("=" * 70)

    # 1. Load Data
    raw_dir = "data/raw"
    os.makedirs(raw_dir, exist_ok=True)
    
    # We sample 25,000 for high-efficiency training while keeping all rare classes
    df_primary = load_primary_dataset(raw_dir=raw_dir, sample_size=25000)
    print(f"Primary dataset loaded: {df_primary.shape}")
    print("Class distribution:\n", df_primary[LABEL_COLUMNS].sum())

    # Load extra dataset for fairness evaluation and bias auditing
    df_extra_bias = load_extra_bias_dataset(raw_dir=raw_dir, sample_size=10000)
    print(f"Extra Unintended Bias dataset loaded: {df_extra_bias.shape}")

    # 2. Stratified Splits
    train_df, val_df, test_df = prepare_stratified_splits(df_primary, test_size=0.15, val_size=0.15)
    
    # Compute positive class weights
    pos_weights = calculate_positive_class_weights(train_df)
    print("\nCalculated Positive Class Weights (inversely proportional to freq):")
    for k, v in pos_weights.items():
        print(f"  {k:15s}: {v:.2f}")

    # 3. Train Fast Baseline (TF-IDF + Balanced Logistic Regression)
    print("\n" + "-" * 50)
    print("TRAINING MODEL 1: TF-IDF + BALANCED LOGISTIC REGRESSION")
    print("-" * 50)
    tfidf_model = TfidfLogisticMultiLabelClassifier(max_features=25000)
    tfidf_model.fit(
        train_df['comment_text'].tolist(),
        train_df[LABEL_COLUMNS].to_numpy(),
        pos_weights
    )
    tfidf_model.save("models/model_artifacts/tfidf_logistic_model.joblib")

    # Evaluate Baseline on Val
    val_probs_tfidf = tfidf_model.predict_proba(val_df['comment_text'].tolist())
    test_probs_tfidf = tfidf_model.predict_proba(test_df['comment_text'].tolist())

    # 4. Train Deep Learning Model (DistilBERT with Class Weighting)
    print("\n" + "-" * 50)
    print("TRAINING MODEL 2: DistilBERT WITH POS_WEIGHT BCE LOSS")
    print("-" * 50)
    distilbert_dir = "models/model_artifacts/distilbert_multilabel"
    
    # Check if DistilBERT is already trained or train it
    if os.path.exists(os.path.join(distilbert_dir, "model.safetensors")) or os.path.exists(os.path.join(distilbert_dir, "pytorch_model.bin")):
        print(f"Loading existing DistilBERT checkpoint from {distilbert_dir}...")
        tokenizer = DistilBertTokenizerFast.from_pretrained(distilbert_dir)
        config = DistilBertConfig.from_pretrained(distilbert_dir)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = DistilBertForMultiLabelClassification.from_pretrained(distilbert_dir, config=config).to(device)
    else:
        # Fine-tune DistilBERT on stratified training set
        model = train_distilbert_model(
            train_texts=train_df['comment_text'].tolist(),
            train_labels=train_df[LABEL_COLUMNS].to_numpy(),
            val_texts=val_df['comment_text'].tolist(),
            val_labels=val_df[LABEL_COLUMNS].to_numpy(),
            pos_weights=pos_weights,
            output_dir=distilbert_dir,
            epochs=2,
            batch_size=32,
            max_len=128
        )
        tokenizer = DistilBertTokenizerFast.from_pretrained(distilbert_dir)

    print("\nRunning inference for Validation and Test splits on DistilBERT...")
    val_probs_distil = predict_distilbert(model, tokenizer, val_df['comment_text'].tolist())
    test_probs_distil = predict_distilbert(model, tokenizer, test_df['comment_text'].tolist())

    # 5. Threshold Tuning (Section 5.3)
    print("\n" + "-" * 50)
    print("TUNING PER-LABEL DECISION THRESHOLDS ON VALIDATION SET")
    print("-" * 50)
    tuning_results = tune_thresholds_for_labels(
        val_df[LABEL_COLUMNS].to_numpy(),
        val_probs_distil
    )
    tuned_thresholds = tuning_results['best_thresholds']

    print("\nComparing Flat 0.5 Cutoff vs Tuned Thresholds on Test Set:")
    comparison = compare_flat_vs_tuned(
        test_df[LABEL_COLUMNS].to_numpy(),
        test_probs_distil,
        tuned_thresholds
    )
    for col, data in comparison['per_label'].items():
        print(f"  {col:13s} | Flat 0.5 F1: {data['flat_0_5']['f1']:.4f} -> Tuned ({data['tuned']['threshold']:.2f}) F1: {data['tuned']['f1']:.4f} (+{data['f1_gain_percent']}%) | AUC: {data['roc_auc']:.4f}")
    print(f"\nMacro F1: Flat = {comparison['macro_summary']['flat_macro_f1']:.4f} -> Tuned = {comparison['macro_summary']['tuned_macro_f1']:.4f} (Absolute Gain: +{comparison['macro_summary']['macro_gain_absolute']:.4f})")

    # Save tuned thresholds and comparisons
    thresholds_payload = {
        'best_thresholds': tuned_thresholds,
        'val_results': tuning_results['val_results'],
        'test_comparison': comparison,
        'curves': tuning_results['curves'],
        'model_name': 'DistilBERT-MultiLabel-WeightedBCE'
    }
    save_thresholds(thresholds_payload, "models/thresholds.json")

    # 6. Full Evaluation Metrics
    eval_metrics = evaluate_predictions(
        test_df[LABEL_COLUMNS].to_numpy(),
        test_probs_distil,
        tuned_thresholds
    )
    with open("models/evaluation_metrics.json", "w") as f:
        json.dump(eval_metrics, f, indent=2)

    # 7. Identity-Term Fairness & Bias Check (Section 5.5)
    print("\n" + "-" * 50)
    print("IDENTITY-TERM BIAS & FAIRNESS AUDIT (Section 5.5)")
    print("-" * 50)
    bias_audit = audit_identity_bias_on_test_set(
        test_df,
        test_probs_distil,
        toxic_threshold=tuned_thresholds.get('toxic', 0.5)
    )
    print(bias_audit['summary_interpretation'])
    
    # Counterfactual probes
    def predict_wrapper(texts):
        return predict_distilbert(model, tokenizer, texts)
    counterfactuals = run_counterfactual_fairness_probes(predict_wrapper)
    
    fairness_results = {
        'audit': bias_audit,
        'counterfactual_probes': counterfactuals
    }
    with open("models/fairness_audit.json", "w") as f:
        json.dump(fairness_results, f, indent=2)

    # 8. Explainability Suite (15–20 Benchmark Test Comments)
    print("\n" + "-" * 50)
    print("GENERATING EXPLAINABILITY SUITE (Section 5.4)")
    print("-" * 50)
    explainer = TokenAttributionExplainer(predict_wrapper)
    
    # Curated diverse test cases covering TPs, TNs, FPs, and edge cases
    benchmark_samples = [
        # True Positives
        {"id": 1, "type": "True Positive (Toxic/Obscene)", "text": "Shut your stupid mouth you absolute idiot, no one cares about your pathetic opinion."},
        {"id": 2, "type": "True Positive (Threat)", "text": "I will find where you live and burn down your house and kill everyone."},
        {"id": 3, "type": "True Positive (Identity Hate)", "text": "All gay and trans people are disgusting subhumans who should be eliminated."},
        {"id": 4, "type": "True Positive (Insult)", "text": "You are the most ignorant and incompetent clown to ever edit this encyclopedia."},
        {"id": 5, "type": "True Positive (Severe Toxic)", "text": "Fuck you and die you filthy piece of trash, go kill yourself right now."},
        # True Negatives (Benign / Civil)
        {"id": 6, "type": "True Negative (Civil Discussion)", "text": "I respectfully disagree with your revision on this section. Could you cite secondary sources?"},
        {"id": 7, "type": "True Negative (Technical)", "text": "Please remember to sign your comments with four tildes on the Wikipedia talk pages."},
        {"id": 8, "type": "True Negative (Neutral)", "text": "The population of Paris was estimated in 2021 based on census data."},
        # Identity Mention Benign (Crucial for van Aken bias audit)
        {"id": 9, "type": "Benign Identity Mention (LGBTQ+)", "text": "I am a proud lesbian woman working in computer science and open-source software."},
        {"id": 10, "type": "Benign Identity Mention (Religion)", "text": "During Ramadan, Muslims fast from dawn to sunset as an act of spiritual discipline."},
        {"id": 11, "type": "Benign Identity Mention (Race)", "text": "The conference celebrates African-American writers and their contributions to literature."},
        {"id": 12, "type": "Benign Identity Mention (Disability)", "text": "We need to ensure that web interfaces are accessible to disabled and blind users."},
        {"id": 13, "type": "Benign Identity Mention (Feminism)", "text": "She was a prominent feminist scholar who advocated for equal wages and education."},
        # Quoting / Meta-Discussion / Subtle Edge Cases
        {"id": 14, "type": "Quoted Meta-Discussion", "text": "The article quotes the politician saying 'they are idiots', but that violates neutral point of view."},
        {"id": 15, "type": "Sarcastic / Heated but Non-abusive", "text": "Oh wonderful, another editor reverting valid citations without checking the talk page first."},
        {"id": 16, "type": "False Positive Risk (Identity Ingestion)", "text": "Gay rights organizations played a key role in advocating for marriage equality laws."},
        {"id": 17, "type": "False Positive Risk (Controversial Topic)", "text": "Jewish culture has deep historical traditions surrounding holiday observances."},
        {"id": 18, "type": "Subtle Threat", "text": "You won't be safe walking to your car tonight if you keep writing these lies."},
        {"id": 19, "type": "Multi-Label Overlap", "text": "Get out of here you filthy bitch, nobody wants you in this group."},
        {"id": 20, "type": "Positive Encouragement", "text": "Thank you so much for clarifying that historical reference, that was exceptionally helpful!"}
    ]

    explanations_output = []
    for item in benchmark_samples:
        txt = item['text']
        exp = explainer.explain_comment(txt, target_label_idx=0)
        item_data = {
            'id': item['id'],
            'category_type': item['type'],
            'text': txt,
            'probabilities': {
                col: round(float(p), 4)
                for col, p in zip(LABEL_COLUMNS, predict_wrapper([txt])[0])
            },
            'explanation': exp
        }
        explanations_output.append(item_data)

    with open("models/benchmark_explanations.json", "w") as f:
        json.dump(explanations_output, f, indent=2)
    print("Saved 20 benchmark explanations to models/benchmark_explanations.json")

    # Save a sample of test data for the interactive batch tester
    sample_test_records = test_df.sample(n=min(200, len(test_df)), random_state=42).to_dict(orient='records')
    with open("data/processed/sample_test_comments.json", "w") as f:
        json.dump(sample_test_records, f, indent=2)

    print("\n" + "=" * 70)
    print("PIPELINE TRAINING & AUDIT COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == '__main__':
    main()
