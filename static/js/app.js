/**
 * Frontend Application Controller for Multi-Label Toxic Comment Classifier
 * Supports:
 * - Light and Dark Mode persistence
 * - Interactive multi-label probability rendering
 * - Token-level explainability (LIME attribution)
 * - Dynamic threshold calibration
 * - Demographic fairness and bias auditing
 */

// Global State
const state = {
  currentTheme: localStorage.getItem('tc_theme') || 'light',
  operatingMode: 'tuned', // 'tuned' or 'flat'
  currentThresholds: {
    toxic: 0.45,
    severe_toxic: 0.25,
    obscene: 0.40,
    threat: 0.20,
    insult: 0.40,
    identity_hate: 0.25
  },
  flatThresholds: {
    toxic: 0.5,
    severe_toxic: 0.5,
    obscene: 0.5,
    threat: 0.5,
    insult: 0.5,
    identity_hate: 0.5
  },
  lastPrediction: null
};

// Preset Examples
const PRESETS = {
  benign: "I respectfully disagree with your revision on this section. Could you cite secondary sources?",
  toxic: "Shut your stupid mouth you absolute idiot, no one cares about your pathetic opinion.",
  threat: "I will find where you live and burn down your house and kill everyone.",
  identity_benign: "I am a proud lesbian woman working in computer science and open-source software.",
  identity_hate: "All gay and trans people are disgusting subhumans who should be eliminated.",
  quoted: "The article quotes the politician saying 'they are idiots', but that violates neutral point of view."
};

const LABEL_CONFIG = {
  toxic: { name: "Toxic", icon: "⚠️", colorClass: "fill-toxic" },
  severe_toxic: { name: "Severe Toxic", icon: "☣️", colorClass: "fill-severe_toxic" },
  obscene: { name: "Obscene", icon: "🔞", colorClass: "fill-obscene" },
  threat: { name: "Threat", icon: "🚨", colorClass: "fill-threat" },
  insult: { name: "Insult", icon: "💢", colorClass: "fill-insult" },
  identity_hate: { name: "Identity Hate", icon: "🛑", colorClass: "fill-identity_hate" }
};

document.addEventListener('DOMContentLoaded', () => {
  initTheme();
  initNavigation();
  initPresets();
  initPredictionUI();
  loadThresholdsData();
  loadFairnessData();
  loadEvaluationData();

  // Run initial analysis with default preset
  document.getElementById('commentInput').value = PRESETS.identity_benign;
  analyzeComment();
});

// ==================== THEME MANAGEMENT ====================
function initTheme() {
  applyTheme(state.currentTheme);

  const toggleBtn = document.getElementById('themeToggleBtn');
  toggleBtn.addEventListener('click', () => {
    state.currentTheme = state.currentTheme === 'light' ? 'dark' : 'light';
    localStorage.setItem('tc_theme', state.currentTheme);
    applyTheme(state.currentTheme);
  });
}

function applyTheme(theme) {
  document.documentElement.setAttribute('data-theme', theme);
  const icon = document.getElementById('themeIcon');
  const label = document.getElementById('themeLabel');
  if (theme === 'dark') {
    icon.textContent = '☀️';
    label.textContent = 'Light Mode';
  } else {
    icon.textContent = '🌙';
    label.textContent = 'Dark Mode';
  }
}

// ==================== NAVIGATION TABS ====================
function initNavigation() {
  const tabBtns = document.querySelectorAll('.tab-btn');
  tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      tabBtns.forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));

      btn.classList.add('active');
      const targetId = btn.getAttribute('data-tab');
      const targetPanel = document.getElementById(targetId);
      if (targetPanel) {
        targetPanel.classList.add('active');
      }
    });
  });
}

// ==================== PRESET BUTTONS ====================
function initPresets() {
  const chips = document.querySelectorAll('.preset-chip');
  chips.forEach(chip => {
    chip.addEventListener('click', () => {
      const presetKey = chip.getAttribute('data-preset');
      if (PRESETS[presetKey]) {
        document.getElementById('commentInput').value = PRESETS[presetKey];
        analyzeComment();
      }
    });
  });

  document.getElementById('clearBtn').addEventListener('click', () => {
    document.getElementById('commentInput').value = '';
    resetPredictionsView();
  });
}

// ==================== PREDICTION & EXPLAINABILITY ====================
function initPredictionUI() {
  const analyzeBtn = document.getElementById('analyzeBtn');
  analyzeBtn.addEventListener('click', analyzeComment);

  // Operating Mode Buttons
  const modeTunedBtn = document.getElementById('modeTunedBtn');
  const modeFlatBtn = document.getElementById('modeFlatBtn');

  modeTunedBtn.addEventListener('click', () => {
    state.operatingMode = 'tuned';
    modeTunedBtn.className = 'btn-primary';
    modeFlatBtn.className = 'btn-secondary';
    if (state.lastPrediction) renderPredictions(state.lastPrediction);
  });

  modeFlatBtn.addEventListener('click', () => {
    state.operatingMode = 'flat';
    modeFlatBtn.className = 'btn-primary';
    modeTunedBtn.className = 'btn-secondary';
    if (state.lastPrediction) renderPredictions(state.lastPrediction);
  });
}

async function analyzeComment() {
  const text = document.getElementById('commentInput').value.trim();
  if (!text) {
    alert("Please enter a comment to analyze.");
    return;
  }

  const analyzeBtn = document.getElementById('analyzeBtn');
  const latencyBadge = document.getElementById('latencyBadge');
  analyzeBtn.disabled = true;
  analyzeBtn.innerHTML = `<span>Evaluating...</span>`;
  latencyBadge.textContent = "Processing...";

  const startTime = performance.now();

  try {
    const response = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: text })
    });

    if (!response.ok) throw new Error("Inference failed.");
    const data = await response.json();
    const elapsed = Math.round(performance.now() - startTime);

    state.lastPrediction = data;
    renderPredictions(data);
    renderExplainability(data.explanation);

    latencyBadge.textContent = `${elapsed}ms • Model: ${data.model_type || 'DistilBERT'}`;
  } catch (err) {
    console.error("Error analyzing comment:", err);
    latencyBadge.textContent = "Inference error";
  } finally {
    analyzeBtn.disabled = false;
    analyzeBtn.innerHTML = `
      <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"></path></svg>
      Analyze Comment
    `;
  }
}

function renderPredictions(data) {
  const probs = data.probabilities;
  const activeThresholds = state.operatingMode === 'tuned' ? state.currentThresholds : state.flatThresholds;
  const container = document.getElementById('labelRows');
  container.innerHTML = '';

  let anyFlagged = false;
  let flaggedLabels = [];

  for (const [key, config] of Object.entries(LABEL_CONFIG)) {
    const prob = probs[key] !== undefined ? probs[key] : 0.0;
    const threshold = activeThresholds[key] !== undefined ? activeThresholds[key] : 0.5;
    const isFlagged = prob >= threshold;
    const pct = Math.min(100, Math.max(0, Math.round(prob * 100)));
    const threshPct = Math.round(threshold * 100);

    if (isFlagged) {
      anyFlagged = true;
      flaggedLabels.push(config.name);
    }

    const row = document.createElement('div');
    row.className = 'label-row';
    row.innerHTML = `
      <div class="label-meta">
        <span class="label-name">
          <span>${config.icon}</span>
          <span>${config.name}</span>
        </span>
        <div class="label-stats">
          <span class="threshold-badge" title="Decision threshold for this category">Cutoff: ${(threshold).toFixed(2)}</span>
          <span class="prob-badge" style="color: ${isFlagged ? 'var(--toxic)' : 'var(--text-primary)'}">
            ${pct}% ${isFlagged ? '<span style="color:var(--toxic); font-size:11px;">[FLAGGED]</span>' : ''}
          </span>
        </div>
      </div>
      <div class="progress-track" title="Probability: ${pct}% | Threshold: ${threshPct}%">
        <div class="progress-fill ${config.colorClass}" style="width: ${pct}%;"></div>
        <div class="threshold-line" style="left: ${threshPct}%;" title="Decision Cutoff (${threshPct}%)"></div>
      </div>
    `;
    container.appendChild(row);
  }

  // Update Overall Banner
  const banner = document.getElementById('overallBanner');
  const bannerIcon = document.getElementById('bannerIcon');
  const bannerTitle = document.getElementById('bannerTitle');
  const bannerSubtitle = document.getElementById('bannerSubtitle');

  if (anyFlagged) {
    banner.className = 'overall-banner toxic';
    bannerIcon.textContent = '🚨';
    bannerTitle.textContent = `Toxicity Detected: ${flaggedLabels.join(', ')}`;
    bannerSubtitle.textContent = `Content breaches operating thresholds in ${flaggedLabels.length} categor${flaggedLabels.length > 1 ? 'ies' : 'y'}. Recommended action: Flag for moderation.`;
  } else {
    banner.className = 'overall-banner safe';
    bannerIcon.textContent = '🛡️';
    bannerTitle.textContent = 'Content Appears Safe';
    bannerSubtitle.textContent = 'All categories remain strictly below active decision thresholds. No harmful violations detected.';
  }
}

function renderExplainability(explanation) {
  const container = document.getElementById('explainContainer');
  const qualBox = document.getElementById('qualitativeBox');
  const qualBadge = document.getElementById('qualitativeBadge');
  const qualText = document.getElementById('qualitativeText');

  if (!explanation) {
    container.innerHTML = '<em>No attribution available.</em>';
    qualBox.style.display = 'none';
    return;
  }

  container.innerHTML = explanation.html || '<em>Token attribution rendered.</em>';

  if (explanation.qualitative_type) {
    qualBox.style.display = 'block';
    qualBadge.textContent = explanation.qualitative_type;

    if (explanation.qualitative_type === 'GENUINE_TOXICITY') {
      qualBadge.className = 'qualitative-badge badge-genuine';
    } else if (explanation.qualitative_type === 'IDENTITY_SENSITIVE') {
      qualBadge.className = 'qualitative-badge badge-identity';
    } else if (explanation.qualitative_type === 'CONTEXTUAL_QUOTING') {
      qualBadge.className = 'qualitative-badge badge-quoting';
    } else {
      qualBadge.className = 'qualitative-badge badge-benign';
    }

    qualText.textContent = explanation.qualitative_note || '';
  } else {
    qualBox.style.display = 'none';
  }
}

function resetPredictionsView() {
  document.getElementById('explainContainer').innerHTML = '<em>Enter a comment to see predictions and token attribution.</em>';
  document.getElementById('qualitativeBox').style.display = 'none';
  document.getElementById('latencyBadge').textContent = 'Ready';
  const banner = document.getElementById('overallBanner');
  banner.className = 'overall-banner safe';
  document.getElementById('bannerTitle').textContent = 'Ready for Analysis';
  document.getElementById('bannerSubtitle').textContent = 'Enter text above to inspect multi-label classifications.';
  document.getElementById('labelRows').innerHTML = '';
}

// ==================== THRESHOLDS DATA & STUDIO ====================
async function loadThresholdsData() {
  try {
    const res = await fetch('/api/thresholds');
    if (!res.ok) return;
    const data = await res.json();

    if (data.best_thresholds) {
      state.currentThresholds = data.best_thresholds;
    }

    // Populate Comparison Table
    if (data.test_comparison && data.test_comparison.per_label) {
      const tbody = document.getElementById('thresholdComparisonTbody');
      tbody.innerHTML = '';
      const prevalences = {
        toxic: '~9.6%',
        severe_toxic: '~1.0%',
        obscene: '~5.3%',
        threat: '~0.3%',
        insult: '~4.9%',
        identity_hate: '~0.9%'
      };

      for (const [col, comp] of Object.entries(data.test_comparison.per_label)) {
        const tr = document.createElement('tr');
        const gainSign = comp.f1_gain_percent >= 0 ? '+' : '';
        tr.innerHTML = `
          <td><strong>${LABEL_CONFIG[col]?.name || col}</strong></td>
          <td>${prevalences[col] || 'N/A'}</td>
          <td>${comp.flat_0_5.f1.toFixed(4)}</td>
          <td><span class="threshold-badge">${comp.tuned.threshold.toFixed(2)}</span></td>
          <td><strong>${comp.tuned.f1.toFixed(4)}</strong></td>
          <td>${comp.roc_auc.toFixed(4)}</td>
          <td class="gain-positive">${gainSign}${comp.f1_gain_percent}%</td>
        `;
        tbody.appendChild(tr);
      }

      if (data.test_comparison.macro_summary) {
        document.getElementById('kpiFlatF1').textContent = data.test_comparison.macro_summary.flat_macro_f1.toFixed(3);
        document.getElementById('kpiTunedF1').textContent = data.test_comparison.macro_summary.tuned_macro_f1.toFixed(3);
        const gainPct = (
          (data.test_comparison.macro_summary.tuned_macro_f1 - data.test_comparison.macro_summary.flat_macro_f1) /
          data.test_comparison.macro_summary.flat_macro_f1 * 100
        ).toFixed(1);
        document.getElementById('kpiGainF1').textContent = `+${gainPct}%`;
      }
    }

    // Build Interactive Sliders
    renderThresholdSliders();
  } catch (err) {
    console.error("Error loading thresholds:", err);
  }
}

function renderThresholdSliders() {
  const container = document.getElementById('slidersGrid');
  container.innerHTML = '';

  for (const [col, config] of Object.entries(LABEL_CONFIG)) {
    const val = state.currentThresholds[col] || 0.40;
    const card = document.createElement('div');
    card.className = 'card';
    card.style.padding = '16px';
    card.innerHTML = `
      <div style="display: flex; justify-content: space-between; align-items: center;">
        <span style="font-weight: 600; font-size: 14px;">${config.icon} ${config.name}</span>
        <span class="slider-val" id="val_${col}">${val.toFixed(2)}</span>
      </div>
      <div class="slider-container">
        <input type="range" class="custom-slider" id="slider_${col}" min="0.05" max="0.90" step="0.01" value="${val}">
      </div>
      <span style="font-size: 11px; color: var(--text-muted);">Swept range: 0.05 - 0.90 (optimal: ${val.toFixed(2)})</span>
    `;
    container.appendChild(card);

    const slider = card.querySelector(`#slider_${col}`);
    const labelVal = card.querySelector(`#val_${col}`);
    slider.addEventListener('input', (e) => {
      const newVal = parseFloat(e.target.value);
      labelVal.textContent = newVal.toFixed(2);
      state.currentThresholds[col] = newVal;
      if (state.lastPrediction && state.operatingMode === 'tuned') {
        renderPredictions(state.lastPrediction);
      }
    });
  }

  document.getElementById('resetThresholdsBtn').addEventListener('click', () => {
    loadThresholdsData();
  });
}

// ==================== FAIRNESS DATA & SANDBOX ====================
async function loadFairnessData() {
  try {
    const res = await fetch('/api/fairness');
    if (!res.ok) return;
    const data = await res.json();
    const audit = data.audit;

    if (audit) {
      if (audit.measurement_a_mean_probabilities) {
        document.getElementById('meanIdProb').textContent = audit.measurement_a_mean_probabilities.comments_with_identity.toFixed(3);
        document.getElementById('meanNonIdProb').textContent = audit.measurement_a_mean_probabilities.comments_without_identity.toFixed(3);
        document.getElementById('idProbRatio').textContent = `${audit.measurement_a_mean_probabilities.ratio.toFixed(2)}x`;
      }

      if (audit.measurement_b_fpr_on_benign) {
        const fprId = (audit.measurement_b_fpr_on_benign.fpr_identity * 100).toFixed(1);
        const fprNonId = (audit.measurement_b_fpr_on_benign.fpr_non_identity * 100).toFixed(1);
        document.getElementById('fprIdRate').textContent = `${fprId}%`;
        document.getElementById('fprNonIdRate').textContent = `${fprNonId}%`;
        document.getElementById('fprDisparityRatio').textContent = `${audit.measurement_b_fpr_on_benign.disparity_ratio.toFixed(1)}x`;
      }

      if (audit.group_breakdown) {
        const tbody = document.getElementById('biasSubgroupTbody');
        tbody.innerHTML = '';
        for (const [group, info] of Object.entries(audit.group_breakdown)) {
          const tr = document.createElement('tr');
          tr.innerHTML = `
            <td><strong>${group}</strong></td>
            <td><span style="font-size: 11px; color: var(--text-muted);">${info.terms.slice(0, 4).join(', ')}...</span></td>
            <td>${info.mean_toxic_prob.toFixed(4)}</td>
            <td><strong>${(info.false_positive_rate * 100).toFixed(1)}%</strong></td>
            <td><span class="threshold-badge" style="color: #ea580c; font-weight: 600;">${info.fpr_ratio_vs_neutral}x neutral</span></td>
          `;
          tbody.appendChild(tr);
        }
      }
    }

    // Counterfactual Sandbox
    initCounterfactualSandbox();
  } catch (err) {
    console.error("Error loading fairness data:", err);
  }
}

function initCounterfactualSandbox() {
  const btn = document.getElementById('runCounterfactualBtn');
  btn.addEventListener('click', async () => {
    const baseText = document.getElementById('counterBaseInput').value.trim();
    const identityTerm = document.getElementById('counterIdentitySelect').value;
    const resultBox = document.getElementById('counterResultBox');

    btn.disabled = true;
    btn.textContent = "Auditing Counterfactual...";

    try {
      const res = await fetch('/api/counterfactual', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ base_text: baseText, identity_term: identityTerm })
      });
      const resData = await res.json();

      resultBox.style.display = 'block';
      const baseProbPct = (resData.baseline_prob * 100).toFixed(1);
      const varProbPct = (resData.variant_prob * 100).toFixed(1);
      const deltaPct = ((resData.variant_prob - resData.baseline_prob) * 100).toFixed(1);
      const sign = deltaPct >= 0 ? '+' : '';

      resultBox.innerHTML = `
        <h4 style="font-size: 14px; font-weight: 600; margin-bottom: 8px;">Counterfactual Audit Results:</h4>
        <div style="font-size: 13px; line-height: 1.8;">
          <div><strong>Neutral Base:</strong> "${resData.baseline_text}" &rarr; <span class="prob-badge" style="color: var(--safe);">${baseProbPct}% toxic</span></div>
          <div><strong>With Identity:</strong> "${resData.variant_text}" &rarr; <span class="prob-badge" style="color: ${resData.is_overflagged ? 'var(--toxic)' : 'var(--primary)'};">${varProbPct}% toxic</span></div>
          <div style="margin-top: 6px; font-size: 12px;">Score Delta: <strong>${sign}${deltaPct}%</strong> ${resData.is_overflagged ? '<span class="qualitative-badge badge-identity" style="margin-left: 8px;">OVERFLAGGING DETECTED</span>' : '<span class="qualitative-badge badge-benign" style="margin-left: 8px;">FAIR / NEUTRAL</span>'}</div>
        </div>
      `;
    } catch (err) {
      console.error(err);
    } finally {
      btn.disabled = false;
      btn.textContent = "Test Counterfactual Pair";
    }
  });
}

// ==================== EVALUATION DATA ====================
async function loadEvaluationData() {
  try {
    const res = await fetch('/api/evaluation');
    if (!res.ok) return;
    const data = await res.json();

    if (data.overall) {
      document.getElementById('kpiMacroAuc').textContent = data.overall.macro_roc_auc.toFixed(3);
      document.getElementById('kpiHamming').textContent = data.overall.hamming_loss.toFixed(4);
      document.getElementById('kpiExactMatch').textContent = `${(data.overall.exact_match_ratio * 100).toFixed(1)}%`;
    }

    if (data.per_label) {
      const tbody = document.getElementById('evaluationTbody');
      tbody.innerHTML = '';
      for (const [col, m] of Object.entries(data.per_label)) {
        const cm = m.confusion_matrix || {};
        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td><strong>${LABEL_CONFIG[col]?.icon || ''} ${LABEL_CONFIG[col]?.name || col}</strong></td>
          <td>${m.precision.toFixed(4)}</td>
          <td>${m.recall.toFixed(4)}</td>
          <td><strong>${m.f1_score.toFixed(4)}</strong></td>
          <td>${m.roc_auc.toFixed(4)}</td>
          <td style="color: var(--safe);">${cm.true_positives || 0}</td>
          <td style="color: var(--toxic);">${cm.false_positives || 0}</td>
          <td style="color: #ea580c;">${cm.false_negatives || 0}</td>
        `;
        tbody.appendChild(tr);
      }
    }
  } catch (err) {
    console.error("Error loading evaluation metrics:", err);
  }
}
