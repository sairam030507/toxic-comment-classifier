"""
Multi-Label Model Architectures:
1. DistilBERTMultiLabelClassifier (PyTorch + Transformers with BCEWithLogitsLoss + pos_weight)
2. TfidfLogisticMultiLabelClassifier (Scikit-Learn fast high-performance ML baseline)
"""

import os
import torch
import torch.nn as nn
import numpy as np
import joblib
from typing import Dict, List, Tuple, Optional, Any
from transformers import DistilBertModel, DistilBertTokenizer, DistilBertPreTrainedModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

LABEL_COLUMNS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']


class DistilBertForMultiLabelClassification(DistilBertPreTrainedModel):
    """
    DistilBERT encoder with multi-label classification head producing 6 logits.
    Uses BCEWithLogitsLoss with positive-class weighting (pos_weight).
    """
    def __init__(self, config, pos_weight: Optional[torch.Tensor] = None):
        super().__init__(config)
        self.num_labels = 6
        self.distilbert = DistilBertModel(config)
        self.pre_classifier = nn.Linear(config.dim, config.dim)
        self.classifier = nn.Linear(config.dim, self.num_labels)
        self.dropout = nn.Dropout(config.seq_classif_dropout)
        self.activation = nn.ReLU()
        
        if pos_weight is not None:
            self.register_buffer('pos_weight', pos_weight)
        else:
            self.pos_weight = None

        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        distilbert_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]   # [CLS] token representation
        pooled_output = self.pre_classifier(pooled_output)
        pooled_output = self.activation(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)  # (bs, 6)

        loss = None
        if labels is not None:
            if self.pos_weight is not None:
                loss_fct = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
            else:
                loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels.float())

        probabilities = torch.sigmoid(logits)

        return {
            'loss': loss,
            'logits': logits,
            'probabilities': probabilities,
            'attentions': distilbert_output.attentions if output_attentions else None,
            'hidden_states': distilbert_output.hidden_states if output_hidden_states else None
        }


class TfidfLogisticMultiLabelClassifier:
    """
    High-performance, ultra-fast baseline using TF-IDF word + char n-grams
    and independent Logistic Regression classifiers with balanced class weighting.
    Provides direct interpretability and word-level coefficient attribution.
    """
    def __init__(self, max_features: int = 25000):
        self.label_cols = LABEL_COLUMNS
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=max_features,
            sublinear_tf=True,
            strip_accents='unicode',
            analyzer='word',
            token_pattern=r'\w{1,}'
        )
        self.classifiers: Dict[str, LogisticRegression] = {}
        self.is_fitted = False

    def fit(self, texts: List[str], y: np.ndarray, class_weights: Optional[Dict[str, float]] = None):
        print("Vectorizing text using TF-IDF (word n-grams 1-3)...")
        X = self.vectorizer.fit_transform(texts)
        
        for idx, col in enumerate(self.label_cols):
            y_col = y[:, idx]
            print(f"Training LogisticRegression for label '{col}' (positives: {int(y_col.sum())}/{len(y_col)})...")
            
            # Use balanced class weighting or custom pos_weight
            clf = LogisticRegression(
                C=2.0,
                max_iter=1000,
                class_weight='balanced',
                solver='lbfgs',
                random_state=42
            )
            clf.fit(X, y_col)
            self.classifiers[col] = clf
            
        self.is_fitted = True
        return self

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """Returns array of shape (N, 6) with predicted probabilities."""
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet!")
        X = self.vectorizer.transform(texts)
        probs = np.zeros((len(texts), len(self.label_cols)))
        for idx, col in enumerate(self.label_cols):
            clf = self.classifiers[col]
            # Probability of positive class (index 1)
            probs[:, idx] = clf.predict_proba(X)[:, 1]
        return probs

    def get_feature_attributions(self, text: str, label_idx: int = 0) -> List[Tuple[str, float]]:
        """
        Calculates exact word contributions for this specific text and target label.
        Weight = TF-IDF(word) * Model_Coefficient(word).
        """
        label_col = self.label_cols[label_idx]
        clf = self.classifiers[label_col]
        X = self.vectorizer.transform([text])
        feature_names = self.vectorizer.get_feature_names_out()
        
        non_zero_indices = X.nonzero()[1]
        contributions = []
        for idx in non_zero_indices:
            word = feature_names[idx]
            tfidf_val = X[0, idx]
            coef = clf.coef_[0, idx]
            score = float(tfidf_val * coef)
            contributions.append((word, score))
            
        # Sort descending by absolute score
        contributions.sort(key=lambda item: abs(item[1]), reverse=True)
        return contributions

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            'vectorizer': self.vectorizer,
            'classifiers': self.classifiers,
            'label_cols': self.label_cols
        }, filepath)
        print(f"Saved TfidfLogisticMultiLabelClassifier to {filepath}")

    @classmethod
    def load(cls, filepath: str):
        data = joblib.load(filepath)
        model = cls()
        model.vectorizer = data['vectorizer']
        model.classifiers = data['classifiers']
        model.label_cols = data['label_cols']
        model.is_fitted = True
        return model
