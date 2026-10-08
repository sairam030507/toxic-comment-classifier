"""
Lightweight Gradio Demonstration Interface (Proposal Section 5.6 & CP4 Deliverable).
Accepts a comment and returns:
- Predicted categories with probabilities
- Active tuned threshold indicators
- Color-coded token attribution highlights
- Qualitative linguistic assessment (slurs vs identity terms vs quotes)
"""

import os
import torch
import numpy as np
import gradio as gr
from typing import Dict, List, Tuple
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.dataset import LABEL_COLUMNS
from src.threshold_tuner import load_thresholds
from src.explainability import TokenAttributionExplainer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "unitary/toxic-bert"

print(f"Loading {MODEL_NAME} for Gradio on {DEVICE}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

thresholds = load_thresholds("models/thresholds.json")

def predict_fn(texts: List[str]) -> np.ndarray:
    clean_texts = [str(t) if str(t).strip() else "empty" for t in texts]
    enc = tokenizer(clean_texts, padding=True, truncation=True, max_length=128, return_tensors='pt').to(DEVICE)
    with torch.no_grad():
        out = model(**enc)
        return torch.sigmoid(out.logits).cpu().numpy()

explainer = TokenAttributionExplainer(predict_fn)


def analyze_comment(text: str, mode: str = "Tuned Thresholds (Recommended)"):
    if not text.strip():
        return {}, "Please enter a comment to analyze.", "No input provided."

    probs = predict_fn([text])[0]
    
    use_tuned = (mode == "Tuned Thresholds (Recommended)")
    active_thresholds = thresholds if use_tuned else {col: 0.5 for col in LABEL_COLUMNS}

    # Format probabilities dictionary for Gradio Label component
    prob_dict = {
        col.replace('_', ' ').title(): float(probs[idx])
        for idx, col in enumerate(LABEL_COLUMNS)
    }

    # Identify flagged categories
    flagged = []
    for idx, col in enumerate(LABEL_COLUMNS):
        t = active_thresholds.get(col, 0.5)
        if probs[idx] >= t:
            flagged.append(f"**{col.replace('_', ' ').title()}** (score: {probs[idx]:.1%}, cutoff: {t:.2f})")

    if flagged:
        summary_md = f"### 🚨 Flagged for Moderation\n" + "\n".join([f"- {f}" for f in flagged])
    else:
        summary_md = "### 🛡️ Content Safe\nAll toxicity categories remain below decision thresholds."

    # Explainability attribution
    explanation = explainer.explain_comment(text, target_label_idx=0)
    
    qual_badge = f"""
    <div style="padding: 12px; border-radius: 8px; background: #f8fafc; border: 1px solid #e2e8f0; margin-top: 8px;">
        <span style="font-weight: 700; font-size: 11px; padding: 2px 8px; border-radius: 9999px; background: #fee2e2; color: #b91c1c;">
            {explanation['qualitative_type']}
        </span>
        <p style="margin-top: 6px; font-size: 13px; color: #475569;">{explanation['qualitative_note']}</p>
        <div style="margin-top: 10px; line-height: 2.0; font-size: 14px;">
            <strong>Token Attribution:</strong><br/>
            {explanation['html']}
        </div>
    </div>
    """

    return prob_dict, summary_md, qual_badge


# Gradio Interface Definition
with gr.Blocks(title="Multi-Label Toxic Comment Classifier | NLP-04") as demo:
    gr.Markdown("""
    # 🛡️ Multi-Label Toxic Comment Classifier
    ### NLP-04 Machine Learning Laboratory • DistilBERT / BERT • Explainability & Fairness
    Multi-label toxicity evaluation across **6 categories**: *Toxic, Severe Toxic, Obscene, Threat, Insult, Identity Hate*.
    """)

    with gr.Row():
        with gr.Column(scale=1):
            input_text = gr.Textbox(
                label="Online Comment Input",
                placeholder="Type or paste any online comment here...",
                lines=4
            )
            mode_select = gr.Radio(
                choices=["Tuned Thresholds (Recommended)", "Flat 0.5 Cutoff"],
                value="Tuned Thresholds (Recommended)",
                label="Decision Threshold Policy"
            )
            submit_btn = gr.Button("Analyze Toxicity & Explain", variant="primary")
            
            gr.Examples(
                examples=[
                    ["I respectfully disagree with your revision on this section. Could you cite secondary sources?"],
                    ["Shut your stupid mouth you absolute idiot, no one cares about your pathetic opinion."],
                    ["I will find where you live and burn down your house and kill everyone."],
                    ["I am a proud lesbian woman working in computer science and open-source software."],
                    ["All gay and trans people are disgusting subhumans who should be eliminated."],
                    ["The article quotes the politician saying 'they are idiots', but that violates neutral point of view."]
                ],
                inputs=[input_text]
            )

        with gr.Column(scale=1):
            output_labels = gr.Label(label="Category Probabilities (0.0 to 1.0)", num_top_classes=6)
            summary_output = gr.Markdown(label="Moderation Decision")
            explain_output = gr.HTML(label="LIME Token Attribution & Qualitative Assessment")

    submit_btn.click(
        fn=analyze_comment,
        inputs=[input_text, mode_select],
        outputs=[output_labels, summary_output, explain_output]
    )

if __name__ == "__main__":
    demo.launch(server_port=7860, theme=gr.themes.Soft(), share=False)
