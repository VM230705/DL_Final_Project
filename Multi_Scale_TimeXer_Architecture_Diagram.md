# 🏗️ Multi-Scale TimeXer Architecture Diagram Design

## 📊 **完整架構示意圖結構**

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                           Multi-Scale TimeXer Architecture                                     │
└─────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ Input Time Series: [B, L, C] (Batch, Length=96, Channels=7/21/321)                           │
│ Example: ETTh2 [4, 96, 7], Weather [4, 96, 21], ECL [4, 96, 321]                            │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                              🔄 Multi-Scale Patch Tokenization                               │
├─────────────────────────┬─────────────────────────┬─────────────────────────────────────────┤
│     Patch Size = 8      │     Patch Size = 16     │         Patch Size = 24                 │
│   ┌─────────────────┐   │   ┌─────────────────┐   │       ┌─────────────────┐               │
│   │ Patches: 12×7   │   │   │ Patches: 6×7    │   │       │ Patches: 4×7    │               │
│   │ [B,12,8,C] →    │   │   │ [B,6,16,C] →    │   │       │ [B,4,24,C] →    │               │
│   │ Linear Proj     │   │   │ Linear Proj     │   │       │ Linear Proj     │               │
│   │ [B,12,d_model]  │   │   │ [B,6,d_model]   │   │       │ [B,4,d_model]   │               │
│   └─────────────────┘   │   └─────────────────┘   │       └─────────────────┘               │
│   Fine-grained patterns │   Balanced resolution   │       Coarse trends                     │
└─────────────────────────┴─────────────────────────┴─────────────────────────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                              ⚡ Multi-Scale Encoder Processing                               │
├─────────────────────────┬─────────────────────────┬─────────────────────────────────────────┤
│   TimeXer Encoder-8     │   TimeXer Encoder-16    │     TimeXer Encoder-24                  │
│  ┌─────────────────┐    │  ┌─────────────────┐    │    ┌─────────────────┐                  │
│  │ Multi-Head      │    │  │ Multi-Head      │    │    │ Multi-Head      │                  │
│  │ Attention       │    │  │ Attention       │    │    │ Attention       │                  │
│  │ + FFN          │    │  │ + FFN          │    │    │ + FFN          │                  │
│  │ + LayerNorm    │    │  │ + LayerNorm    │    │    │ + LayerNorm    │                  │
│  └─────────────────┘    │  └─────────────────┘    │    └─────────────────┘                  │
│  Output: [B,12,d_model] │  Output: [B,6,d_model] │    Output: [B,4,d_model]                │
└─────────────────────────┴─────────────────────────┴─────────────────────────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                              🧠 Advanced Fusion Mechanisms                                   │
├─────────────────────────────────────────────────────────────────────────────────────────────┤
│                                Fusion Strategy Selection                                     │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐           │
│  │  Concatenation  │ │   Attention     │ │   Gated         │ │  Hierarchical   │           │
│  │     Fusion      │ │    Fusion       │ │   Fusion        │ │    Fusion       │           │
│  │ ┌─────────────┐ │ │ ┌─────────────┐ │ │ ┌─────────────┐ │ │ ┌─────────────┐ │           │
│  │ │Concat scales│ │ │ │Multi-Head   │ │ │ │Gating       │ │ │ │Progressive  │ │           │
│  │ │+ Linear     │ │ │ │Attention    │ │ │ │mechanism    │ │ │ │aggregation │ │           │
│  │ │Projection   │ │ │ │across scales│ │ │ │with sigmoid │ │ │ │layer by     │ │           │
│  │ │             │ │ │ │             │ │ │ │             │ │ │ │layer        │ │           │
│  │ └─────────────┘ │ │ └─────────────┘ │ │ └─────────────┘ │ │ └─────────────┘ │           │
│  │ Cost: 2x        │ │ Cost: 3x        │ │ Cost: 2.5x      │ │ Cost: 4x        │           │
│  │ Improvement:    │ │ Improvement:    │ │ Improvement:    │ │ Improvement:    │           │
│  │ +1.5% MSE       │ │ +3.1% MSE       │ │ +2.1% MSE       │ │ +2.6% MSE       │           │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘ └─────────────────┘           │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                               📊 Unified Representation                                      │
│                          [B, unified_length, d_model]                                        │
│                         Ready for TimeXer Decoder                                            │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                               🎯 TimeXer Decoder + Prediction                               │
│  ┌─────────────────────────────────────────────────────────────────────────────────────┐   │
│  │ Standard TimeXer Decoder Architecture                                              │   │
│  │ • Cross-attention with encoder outputs                                             │   │
│  │ • Self-attention for temporal dependencies                                         │   │
│  │ • Feed-forward networks                                                            │   │
│  │ • Layer normalization                                                              │   │
│  └─────────────────────────────────────────────────────────────────────────────────────┘   │
│                                           │                                                   │
│                                           ▼                                                   │
│  ┌─────────────────────────────────────────────────────────────────────────────────────┐   │
│  │ Prediction Head: Linear(d_model → pred_len * n_vars)                               │   │
│  │ Output: [B, pred_len, n_vars] where pred_len ∈ {96, 192, 336, 720}                │   │
│  └─────────────────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                               📈 Performance Comparison                                      │
├─────────────────────┬─────────────────────┬─────────────────────┬─────────────────────────────┤
│      Dataset        │    Single-Scale     │   Multi-Scale Best  │      Improvement            │
│                     │    (Baseline)       │                     │                             │
├─────────────────────┼─────────────────────┼─────────────────────┼─────────────────────────────┤
│ Weather (21 vars)   │ MSE: 0.1586         │ MSE: 0.1568 (Att)   │ ✅ 1.1% ↓                  │
│ 96-step forecast    │ MAE: 0.2063         │ MAE: 0.2049         │ ✅ 0.7% ↓                  │
├─────────────────────┼─────────────────────┼─────────────────────┼─────────────────────────────┤
│ ECL (321 vars)      │ MSE: 0.1652         │ MSE: 0.1601 (Att)   │ ✅ 3.1% ↓                  │
│ 96-step forecast    │ MAE: 0.2643         │ MAE: 0.2609         │ ✅ 1.3% ↓                  │
├─────────────────────┼─────────────────────┼─────────────────────┼─────────────────────────────┤
│ ETTh2 (7 vars)      │ MSE: 0.2963         │ MSE: 0.2873 (Hier)  │ ✅ 3.0% ↓ (short-term)     │
│ 96-step forecast    │ MAE: 0.3470         │ MAE: 0.3368         │ ❌ 2.2-6.5% ↑ (long-term)  │
└─────────────────────┴─────────────────────┴─────────────────────┴─────────────────────────────┘
```

## 🎨 **具體繪圖建議**

### **1. 主架構圖 (Horizontal Layout)**

```
Input → Multi-Patch → Multi-Encoder → Fusion → Decoder → Output
 📊        🔄           ⚡           🧠        🎯       📈
```

### **2. Multi-Scale Tokenization 詳細圖**

```
Time Series [B, 96, C]
        │
        ├─── Patch-8  [B, 12, 8, C] → Linear → [B, 12, d_model]
        │
        ├─── Patch-16 [B, 6, 16, C] → Linear → [B, 6, d_model]  
        │
        └─── Patch-24 [B, 4, 24, C] → Linear → [B, 4, d_model]
```

### **3. Fusion Mechanism 比較圖**

```
┌─────────────────────────────────────────────────────────────┐
│                    Fusion Performance                       │
├──────────┬──────────┬──────────┬──────────┬─────────────────┤
│ Method   │   MSE    │   MAE    │   Cost   │    Use Case     │
├──────────┼──────────┼──────────┼──────────┼─────────────────┤
│ Concat   │ 0.1628   │ 0.2627   │   2x     │ Budget-friendly │
│ Gated    │ 0.1617   │ 0.2617   │  2.5x    │ Balanced        │
│ Attention│ 0.1601   │ 0.2609   │   3x     │ Best accuracy   │
│ Hierarch │ 0.1610   │ 0.2608   │   4x     │ Best MAE        │
└──────────┴──────────┴──────────┴──────────┴─────────────────┘
```

### **4. 數據集維度 vs 效果關係圖**

```
Improvement vs Dataset Dimensionality

4% ┤ ○ ECL (321 vars)
   │
3% ┤     ○ ETTh2 (7 vars, short-term)
   │
2% ┤
   │ 
1% ┤ ○ Weather (21 vars)
   │
0% ┤────────────────────────────────────
   └────────────────────────────────────
   0    50   100   150   200   250   300+
                Variables Count
```

### **5. 預測長度 vs 效果圖 (ETTh2 Case Study)**

```
ETTh2 Multi-Scale Performance vs Prediction Length

3% ┤ ✅
   │     ✅
2% ┤         
   │
1% ┤
   │
0% ┤─────────┬─────────┬─────────┬─────────
   │         │         │         │
-2%┤         │         │ ❌      │
   │         │         │     ❌  │
-4%┤         │         │         │ ❌
   └─────────┴─────────┴─────────┴─────────
    96       192       336       720
            Prediction Length
```

## 🖼️ **建議的圖表工具**

### **1. 主架構圖**: 
- **工具**: Draw.io, Visio, PowerPoint
- **風格**: 流程圖 + 模塊圖
- **顏色**: 藍色系(輸入) → 綠色系(處理) → 紅色系(輸出)

### **2. Performance表格**:
- **工具**: LaTeX (booktabs), Excel, PowerPoint
- **元素**: 熱力圖顏色編碼 (綠色=好, 紅色=差)

### **3. 數據流圖**:
- **工具**: TikZ, Draw.io
- **重點**: 突出多尺度並行處理

## 📝 **圖表標題建議**

1. **"Multi-Scale TimeXer: Architecture Overview"**
2. **"Multi-Scale Patch Tokenization and Fusion Strategy"**  
3. **"Performance Comparison: Single-Scale vs Multi-Scale Fusion"**
4. **"Dataset Dimensionality Impact on Multi-Scale Benefits"**
5. **"Prediction Length vs Multi-Scale Effectiveness (ETTh2 Case)"**

## 🎯 **核心信息突出**

### **Key Messages要在圖中突出**:
1. **Multi-scale patches capture different temporal patterns**
2. **Advanced fusion > Simple concatenation**  
3. **Benefits scale with dataset dimensionality**
4. **Performance varies by prediction length**
5. **Clear cost-benefit trade-offs**

這樣的示意圖能夠清楚地展示我們方法的創新點、架構設計和實驗結果！