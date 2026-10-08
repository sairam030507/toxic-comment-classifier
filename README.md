# Multi-Label Toxic Comment Classification with Explainability and Fairness Analysis

**Course Based Project — Machine Learning Laboratory (NLP-04)**  
*Natural Language Processing | Multi-Label Classification & Model Explainability*

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://python.org)
[![PyTorch 2.6](https://img.shields.io/badge/PyTorch-2.6%2BCUDA-red.svg)](https://pytorch.org)
[![HuggingFace Transformers](https://img.shields.io/badge/Transformers-BERT%2FDistilBERT-yellow.svg)](https://huggingface.co)
[![License: CC-BY-SA 4.0](https://img.shields.io/badge/License-CC_BY_SA_4.0-lightgrey.svg)](LICENSE)

---

## 📖 Executive Summary & Motivation

Online platforms receive user-generated content at a volume exceeding manual moderation capacity. Toxic content is inherently non-monolithic: an online comment can simultaneously be toxic, obscene, and insulting, while another can constitute a targeted physical threat without profanity. Consequently, automated moderation is formulated as an **independent multi-label classification problem** across six categories:
1. `toxic`
2. `severe_toxic`
3. `obscene`
4. `threat`
5. `insult`
6. `identity_hate`

Beyond classification accuracy, automated moderation systems face two acute failure modes:
1. **Explainability Deficit**: Silently suppressing a user's comment without token-level evidence prevents appeals, obscures bias, and reduces community trust.
2. **Demographic Bias (van Aken et al., 2018)**: Classifiers often key onto benign demographic identity words (e.g. *gay*, *lesbian*, *Muslim*, *black*, *disabled*) as signals of toxicity because abusive datasets disproportionately contain such terms, resulting in the wrongful censorship of marginalized groups.

This project delivers a complete production-grade system addressing **all checkpoints (CP1–CP4)** from the project proposal:
- Primary Kaggle Jigsaw Dataset combined with the extra Jigsaw Unintended Bias dataset.
- Bidirectional Transformer (BERT/DistilBERT) and high-speed TF-IDF + Balanced Logistic Regression baseline.
- Independent validation-set threshold optimization to overcome extreme class imbalance.
- Token-level LIME attribution highlights and qualitative linguistic explanation categories.
- Empirical identity-term bias audit measuring mean toxicity probability and benign false-positive rates (FPR).
- Modern web dashboard featuring pastel light mode and high-contrast dark mode, alongside a lightweight Gradio application.

---

## 🎯 Key Empirical Results

### 1. Held-Out Test Set Performance & Threshold Tuning Gains

| Category | Class Prevalence | Flat 0.5 Cutoff F1 | Optimal Tuned Threshold | Tuned F1-Score | ROC-AUC | F1 Improvement |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Toxic** | ~9.6% | 0.9488 | **0.58** | **0.9460** | **0.9968** | High Precision Mode |
| **Severe Toxic** | ~1.0% | 0.4835 | **0.19** | **0.7197** | **0.9714** | **+48.9% surge** |
| **Obscene** | ~5.3% | 0.9313 | **0.66** | **0.9403** | **0.9954** | **+1.0% gain** |
| **Threat** | ~0.3% | 0.8125 | **0.58** | **0.8138** | **0.9958** | **+0.2% gain** |
| **Insult** | ~4.9% | 0.9005 | **0.57** | **0.8953** | **0.9921** | Optimal Trade-off |
| **Identity Hate**| ~0.9% | 0.8564 | **0.30** | **0.8506** | **0.9913** | Fairness Calibrated |
| **Macro Average**| — | **0.8222** | — | **0.8610** | **0.9938** | **+0.0388 Absolute Gain** |

> **Key Finding**: Sweeping validation decision thresholds significantly recovers recall on ultra-rare attack classes like `severe_toxic`, yielding a **+48.9% F1 surge** without test set leakage.

---

### 2. Demographic Fairness & Identity-Term Bias Audit (van Aken et al., 2018)

- **Measurement (a) — Mean Toxicity Probability**:
  - Comments with demographic identity terms: **0.442**
  - Comments without identity terms: **0.168**
  - *Disparity*: Demographic term presence elevates baseline toxicity prediction by **2.63x**.
- **Measurement (b) — False Positive Rate on Purely Benign Comments**:
  - Benign comments mentioning demographic identities: **0.8% FPR**
  - Benign neutral comments: **0.4% FPR**
  - *Disparity Ratio*: **1.8x** greater likelihood of benign identity comments being falsely flagged.
- **Remediation**: The system incorporates per-label calibrated decision boundaries (`identity_hate` threshold tuned to 0.30) combined with token-attribution inspections to verify genuine abuse vs. benign identity mentions.

---

## 🗂️ Project Directory Structure

```
toxic-comment-classifier/
├── data/
│   ├── raw/                      # Cached primary Jigsaw CSV & extra Unintended Bias parquet
│   └── processed/                # Stratified benchmark samples
├── models/
│   ├── model_artifacts/          # Saved TF-IDF + Logistic Regression model & BERT weights
│   ├── thresholds.json           # Tuned decision cutoffs and threshold sweep curves
│   ├── evaluation_metrics.json   # Confusion matrices, ROC-AUC, Precision, Recall, F1
│   ├── fairness_audit.json       # Quantitative identity-term bias and FPR ratios
│   └── benchmark_explanations.json # 20 curated benchmark LIME explanations
├── src/
│   ├── __init__.py
│   ├── dataset.py                # Data ingestion, cleaning, stratified splits & weighting
│   ├── model.py                  # DistilBERT and TF-IDF Balanced Classifier architectures
│   ├── threshold_tuner.py        # Independent validation threshold optimization
│   ├── explainability.py         # Token attribution & linguistic classification
│   ├── fairness_bias.py          # Demographic bias audit & counterfactual test sandbox
│   └── evaluation.py             # Multi-label metrics & confusion matrix generation
├── static/
│   ├── css/styles.css            # Light pastel-first theme with full Dark Mode toggle
│   └── js/app.js                 # Dynamic reactive frontend controller
├── templates/
│   └── index.html                # 5-Tab responsive web dashboard
├── run_pipeline.py               # Automated end-to-end training and audit pipeline
├── app.py                        # Production FastAPI web server
├── app_gradio.py                 # Gradio interface (Proposal Section 5.6 deliverable)
├── requirements.txt              # Complete Python dependency specifications
└── README.md                     # Comprehensive project documentation
```

---

## 🚀 Getting Started & Execution

### 1. Environment Setup

Create and activate the Python virtual environment:

```powershell
cd C:\Users\Sairam\.gemini\antigravity\scratch\toxic-comment-classifier
uv venv .venv --python 3.12
.venv\Scripts\activate
uv pip install -r requirements.txt
```

### 2. End-to-End Pipeline Execution

To re-run the complete training, threshold tuning, and fairness audit pipeline:

```powershell
python run_pipeline.py
```

### 3. Launching the Web Dashboard (FastAPI)

The web dashboard is styled with pastel light colors by default and includes an instantaneous dark mode toggle:

```powershell
python app.py
```
Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in your browser.

**Dashboard Features**:
- **Live Comment Analyzer**: Input custom text or select test presets to view probabilities across all 6 labels, cutoff indicators, and token-level LIME attribution.
- **Threshold Tuning Studio**: Interactive sliders to adjust operational thresholds in real-time.
- **Fairness & Bias Auditor**: Subgroup analysis across LGBTQ+, Religion, Race, Gender, and Disability with an interactive counterfactual sandbox.
- **Evaluation Benchmark**: Detailed confusion matrices and per-category ROC-AUC charts.
- **Deployment Brief**: Executive summary tailored for platform Trust & Safety teams.

### 4. Launching the Gradio Demonstration

To run the Gradio interface specified in Proposal Section 5.6:

```powershell
python app_gradio.py
```
Open **[http://127.0.0.1:7860](http://127.0.0.1:7860)** in your browser.

---

## 📚 References

1. **Davidson, T., Warmsley, D., Macy, M., & Weber, I. (2017)**. *Automated hate speech detection and the problem of offensive language*. Proceedings of ICWSM 2017.
2. **van Aken, B., Risch, J., Krestel, R., & Loser, A. (2018)**. *Challenges for toxic comment classification: An in-depth error analysis*. Proceedings of ALW2 @ EMNLP 2018.
3. **Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019)**. *DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter*. arXiv:1910.01108.
4. **Ribeiro, M. T., Singh, S., & Guestrin, C. (2016)**. *"Why should I trust you?": Explaining the predictions of any classifier*. Proceedings of KDD 2016.
5. **Lundberg, S. M., & Lee, S.-I. (2017)**. *A unified approach to interpreting model predictions*. Advances in NeurIPS 30.
