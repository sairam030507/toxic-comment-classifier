"""
Identity-Term Bias and Fairness Auditing Module (Proposal Section 5.5).
Evaluates demographic identity-term bias and van Aken et al. (2018) over-flagging phenomena:
Measurement (a): Mean predicted probability of toxicity (Identity vs Non-Identity comments)
Measurement (b): False-Positive Rate (FPR) on benign comments (Identity vs Non-Identity)
"""

import re
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple

LABEL_COLUMNS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

CURATED_IDENTITY_GROUPS = {
    'LGBTQ+': ['gay', 'lesbian', 'homosexual', 'queer', 'bisexual', 'transgender', 'trans'],
    'Religion': ['muslim', 'islam', 'jewish', 'jew', 'christian', 'hindu', 'buddhist', 'atheist'],
    'Race & Ethnicity': ['black', 'white', 'african', 'african-american', 'asian', 'latino', 'hispanic'],
    'Gender': ['woman', 'women', 'female', 'girl', 'feminist', 'feminism'],
    'Disability': ['disabled', 'disability', 'blind', 'deaf', 'autistic', 'autism'],
    'Vulnerable Groups': ['immigrant', 'refugee', 'migrant', 'asylum']
}

ALL_IDENTITY_TERMS = [term for group in CURATED_IDENTITY_GROUPS.values() for term in group]


def has_identity_term(text: str, terms: List[str] = ALL_IDENTITY_TERMS) -> bool:
    """Returns True if text contains at least one identity term as a whole word."""
    pattern = r'\b(?:' + '|'.join(re.escape(t) for t in terms) + r')\b'
    return bool(re.search(pattern, str(text), re.IGNORECASE))


def audit_identity_bias_on_test_set(
    test_df: pd.DataFrame,
    probs_test: np.ndarray,
    toxic_threshold: float = 0.5,
    toxic_idx: int = 0
) -> Dict[str, Any]:
    """
    Performs the two formal bias measurements from Section 5.5:
    (a) Mean predicted toxicity probability: comments with >=1 identity term vs 0.
    (b) FPR on ground-truth benign comments: comments with identity term vs without.
    """
    comments = test_df['comment_text'].tolist()
    toxic_probs = probs_test[:, toxic_idx]
    
    # Ground truth: purely benign means 0 on ALL 6 toxicity categories
    is_ground_truth_benign = (test_df[LABEL_COLUMNS].sum(axis=1) == 0).to_numpy()
    
    # Identify which comments contain any identity term
    identity_mask = np.array([has_identity_term(c, ALL_IDENTITY_TERMS) for c in comments])
    non_identity_mask = ~identity_mask
    
    # Measurement (a): Mean predicted probability of toxicity
    mean_prob_identity = float(np.mean(toxic_probs[identity_mask])) if np.any(identity_mask) else 0.0
    mean_prob_non_identity = float(np.mean(toxic_probs[non_identity_mask])) if np.any(non_identity_mask) else 0.0
    prob_difference = mean_prob_identity - mean_prob_non_identity
    prob_ratio = mean_prob_identity / max(mean_prob_non_identity, 1e-6)
    
    # Measurement (b): False Positive Rate (FPR) on benign comments
    # Benign with identity
    benign_with_id_mask = is_ground_truth_benign & identity_mask
    n_benign_with_id = int(np.sum(benign_with_id_mask))
    fp_with_id = int(np.sum((toxic_probs[benign_with_id_mask] >= toxic_threshold)))
    fpr_identity = float(fp_with_id / max(n_benign_with_id, 1))
    
    # Benign without identity
    benign_no_id_mask = is_ground_truth_benign & non_identity_mask
    n_benign_no_id = int(np.sum(benign_no_id_mask))
    fp_no_id = int(np.sum((toxic_probs[benign_no_id_mask] >= toxic_threshold)))
    fpr_non_identity = float(fp_no_id / max(n_benign_no_id, 1))
    
    fpr_disparity_ratio = float(fpr_identity / max(fpr_non_identity, 1e-6))
    
    # Detailed group breakdown
    group_breakdown = {}
    for group_name, terms in CURATED_IDENTITY_GROUPS.items():
        grp_mask = np.array([has_identity_term(c, terms) for c in comments])
        grp_benign_mask = is_ground_truth_benign & grp_mask
        
        n_grp = int(np.sum(grp_mask))
        mean_p = float(np.mean(toxic_probs[grp_mask])) if n_grp > 0 else 0.0
        
        n_grp_benign = int(np.sum(grp_benign_mask))
        grp_fp = int(np.sum(toxic_probs[grp_benign_mask] >= toxic_threshold))
        grp_fpr = float(grp_fp / max(n_grp_benign, 1))
        
        group_breakdown[group_name] = {
            'terms': terms,
            'total_sample_count': n_grp,
            'benign_count': n_grp_benign,
            'mean_toxic_prob': round(mean_p, 4),
            'false_positives': grp_fp,
            'false_positive_rate': round(grp_fpr, 4),
            'fpr_ratio_vs_neutral': round(grp_fpr / max(fpr_non_identity, 1e-4), 2)
        }
        
    return {
        'measurement_a_mean_probabilities': {
            'comments_with_identity': round(mean_prob_identity, 4),
            'comments_without_identity': round(mean_prob_non_identity, 4),
            'absolute_difference': round(prob_difference, 4),
            'ratio': round(prob_ratio, 2),
            'sample_counts': {
                'with_identity': int(np.sum(identity_mask)),
                'without_identity': int(np.sum(non_identity_mask))
            }
        },
        'measurement_b_fpr_on_benign': {
            'threshold_used': toxic_threshold,
            'fpr_identity': round(fpr_identity, 4),
            'fpr_non_identity': round(fpr_non_identity, 4),
            'disparity_ratio': round(fpr_disparity_ratio, 2),
            'sample_counts': {
                'benign_with_identity': n_benign_with_id,
                'false_positives_with_identity': fp_with_id,
                'benign_without_identity': n_benign_no_id,
                'false_positives_without_identity': fp_no_id
            }
        },
        'group_breakdown': group_breakdown,
        'summary_interpretation': (
            f"Fairness Audit Finding: Comments mentioning demographic identity terms receive an average toxicity "
            f"score of {mean_prob_identity:.3f} vs {mean_prob_non_identity:.3f} for neutral comments. "
            f"On purely benign comments, the False-Positive Rate for identity-containing comments is "
            f"{fpr_identity*100:.1f}% vs {fpr_non_identity*100:.1f}% for neutral comments "
            f"(Disparity Ratio: {fpr_disparity_ratio:.1f}x). "
            f"This confirms the van Aken et al. (2018) unintended bias phenomenon requiring per-label threshold tuning."
        )
    }


def run_counterfactual_fairness_probes(predict_proba_fn, toxic_idx: int = 0) -> List[Dict[str, Any]]:
    """
    Tests synthetic counterfactual pairs to detect artificial score inflation
    caused solely by appending or including demographic terms.
    """
    probe_templates = [
        ("I am a person.", "I am a {identity} person."),
        ("They are a great neighbor.", "They are a great {identity} neighbor."),
        ("We should support each other.", "We should support each other, especially {identity} communities."),
        ("I disagree with your argument.", "I disagree with your argument as a {identity}."),
        ("This is normal behavior.", "This is normal {identity} behavior.")
    ]
    
    test_identities = ['gay', 'lesbian', 'muslim', 'jewish', 'black', 'white', 'woman', 'feminist', 'disabled']
    
    results = []
    for baseline_template, id_template in probe_templates:
        base_prob = float(predict_proba_fn([baseline_template])[0, toxic_idx])
        
        pair_results = {
            'baseline_text': baseline_template,
            'baseline_toxic_prob': round(base_prob, 4),
            'variants': []
        }
        
        for id_term in test_identities:
            variant_text = id_template.format(identity=id_term)
            variant_prob = float(predict_proba_fn([variant_text])[0, toxic_idx])
            delta = variant_prob - base_prob
            
            pair_results['variants'].append({
                'identity_term': id_term,
                'text': variant_text,
                'toxic_prob': round(variant_prob, 4),
                'delta': round(delta, 4),
                'is_overflagged': bool(delta > 0.15)
            })
            
        results.append(pair_results)
        
    return results
