# 📘 TimeXer-Based Time Series Forecasting: Detailed Improvement Plan

---

## 📌 Project Goal

Enhance the TimeXer architecture to improve performance and robustness in:
- Diverse temporal patterns (short- and long-term)
- High-dimensional datasets (e.g., Traffic)
- Spike prediction (sudden value changes)
- Dynamic exogenous variable fusion
- Cross-variable interaction learning

We will build upon the original [TimeXer GitHub repo](https://github.com/thuml/TimeXer) and progressively enhance its architecture and training process.

---

## 🎯 Motivation

Although TimeXer achieves competitive results using fixed-length patches, its use of a **uniform patch length = 16 for all long-term forecasting datasets** (e.g., ETT, Traffic, Weather, Electricity) limits flexibility. In our pre-experiments, we observed that:
- Patch length = 8 or 24 sometimes outperformed 16 on specific datasets.
- No single patch size works best across all datasets or prediction lengths.
- TimeXer does not attempt to **adapt** patch resolution.

Additionally:
- No channel selection: leads to poor performance on datasets like Traffic (862 channels).
- Poor spike prediction: MAE may look fine, but MSE is large due to occasional spikes.
- Exogenous variables are fused only via a global token, ignoring finer dynamics.
- Variable-to-variable relations are not modeled (unlike iTransformer).

These motivate our modular improvements below.

---

## 📊 Dataset Characteristics Analysis

Based on our dataset analysis, we identified the following characteristics:

### Dataset Scale Classification
- **Small-scale (< 10 features)**: ETTh1, ETTh2, ETTm1, ETTm2, Exchange
- **Medium-scale (10-100 features)**: Weather (21 features)
- **Large-scale (> 100 features)**: Traffic (862 features), ECL (321 features)

### Time Series Length Classification
- **Short series (< 20k steps)**: ETTh1, ETTh2, Traffic, Exchange
- **Medium series (20k-50k steps)**: ECL, Weather  
- **Long series (> 50k steps)**: ETTm1, ETTm2

### Key Insights for Architecture Design
1. **Multi-scale approach needed**: Different datasets have vastly different feature dimensions (8 vs 862)
2. **Channel selection critical**: Large datasets like Traffic need selective attention
3. **Temporal resolution varies**: Different time granularities (hourly vs 15-min vs 10-min)
4. **Data distribution diversity**: Each dataset has unique statistical properties

---

## 🛠️ Proposed Modules

### ✅ M1. Multi-Scale Patch Tokenization
- Use multiple patch sizes (e.g., 8, 16, 24)
- Fuse multi-resolution temporal tokens
- **Motivation**: Adapt to different temporal patterns in various datasets

### ✅ M2. Learnable Patch Fusion
- Attention/gating-based fusion of tokens from different patch scales
- **Motivation**: Learn optimal combination of multi-scale features

### ✅ M3. Channel Masking / Gating
- SE-block-style module to suppress uninformative channels
- **Motivation**: Critical for high-dimensional datasets like Traffic (862 channels)

### ✅ M4. Spike-aware Loss Function
- Add loss weight to high-error or high-derivative regions
- **Motivation**: Better capture sudden value changes and extreme events

### ✅ M5. Enhanced Exogenous Fusion
- Apply exo-to-patch cross attention at multiple encoder layers
- **Motivation**: Better utilize temporal context features

### ✅ M6. Channel-Wise Attention & Fusion Head
- Add iTransformer-style variate fusion to final encoder output
- **Motivation**: Model cross-variable interactions

---

## 📊 Baseline: TimeXer Results on Datasets (Patch Length = 16)

| Dataset     | Horizon | MAE   | MSE   | Features | Time Steps |
|-------------|---------|-------|-------|----------|------------|
| ETTm2       | 96      | 0.188 | 0.051 | 7        | 69,680     |
| ETTh1       | 96      | 0.384 | 0.300 | 7        | 17,420     |
| Weather     | 96      | 0.158 | 0.035 | 21       | 52,696     |
| Traffic     | 96      | 0.418 | 0.281 | 862      | 17,544     |
| Electricity | 96      | 0.127 | 0.026 | 321      | 26,304     |

> Source: TimeXer Appendix G. For long-term forecasting tasks, PL=16 is used with input length = 96.

---

## 🔬 Experimental Setup

### Datasets
- **Small-scale**: ETTh1, ETTm2 (7 features, various lengths)
- **Medium-scale**: Weather (21 features, 52k steps)
- **Large-scale**: Traffic (862 features), ECL (321 features)

### Baselines
- TimeXer (PL=16)
- iTransformer
- PatchTST

### Ablation Studies
| Variant     | Modules Active             | Target Datasets           |
|-------------|----------------------------|---------------------------|
| TimeXer     | Baseline only              | All                       |
| +M1         | Multi-scale patch only     | All                       |
| +M1 + M3    | Patch + Channel Mask       | Traffic, ECL (high-dim)   |
| +M1 + M4    | Patch + Spike-aware Loss   | All (focus on extremes)   |
| +All        | Full model (M1~M6)         | All                       |

---

## 🧪 Development Phases

### Phase 1: Reproduce Baseline ✅
- Run TimeXer with PL=16
- Validate logging and metrics
- Complete dataset analysis

### Phase 2: Integrate Modules
1. Add M1 (multi-patch tokenizer) - **Priority: High** 
2. Add M3 (channel masking) - **Priority: High for Traffic/ECL**
3. Add M4 (spike-aware loss) - **Priority: Medium**
4. Add M5 (exo fusion) - **Priority: Medium**
5. Add M6 (channel-wise fusion) - **Priority: Low**

### Phase 3: Benchmark + Ablation
- Compare accuracy, spike error
- Visualize learned masks/weights
- Dataset-specific analysis

### Phase 4: Final Report
- Table/chart output
- Model diagram
- Codebase release

---

## 📦 Deliverables

- Enhanced TimeXer repository
- Dataset analysis report ✅
- Experiment logs + tables
- Final report
- Presentation slides

---

## ⭐ Summary

This project upgrades TimeXer by introducing adaptivity in temporal resolution, channel selection, exogenous fusion, spike sensitivity, and variate reasoning. The dataset analysis reveals the need for different strategies for small-scale (ETT, Exchange), medium-scale (Weather), and large-scale (Traffic, ECL) datasets, motivating our modular approach that combines the best of TimeXer, Pathformer, and iTransformer.
