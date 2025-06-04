```
📊 Multi-Scale TimeXer Architecture Diagram

Input Time Series: [B, seq_len=96, n_vars]
                    ↓
        ┌─────────────────────────────────────┐
        │     Multi-Scale Patching            │
        └─────────────────────────────────────┘
                    ↓
    ┌───────────┬───────────┬───────────┐
    │ Patch_8   │ Patch_16  │ Patch_24  │
    │ 12 tokens │ 6 tokens  │ 4 tokens  │
    └───────────┴───────────┴───────────┘
                    ↓
        ┌─────────────────────────────────────┐
        │     Independent Embedding           │
        │  patch_size → d_model (256)         │
        └─────────────────────────────────────┘
                    ↓
    ┌───────────┬───────────┬───────────┐
    │[B,V,13,D] │[B,V,7,D]  │[B,V,5,D]  │
    │+global tok│+global tok│+global tok│
    └───────────┴───────────┴───────────┘
                    ↓
        ┌─────────────────────────────────────┐
        │        Fusion Module                │
        │   (Attention/Gated/Scale-Aware)     │
        └─────────────────────────────────────┘
                    ↓
            [B, n_vars, 25, d_model]
            (13+7+5=25 total tokens)
                    ↓
        ┌─────────────────────────────────────┐
        │     Reshape for Encoder             │
        │   [B*n_vars, 25, d_model]           │
        └─────────────────────────────────────┘
                    ↓
        ┌─────────────────────────────────────┐
        │      Transformer Encoder            │
        │    (Self-Attention + Cross-Attn)    │
        └─────────────────────────────────────┘
                    ↓
        ┌─────────────────────────────────────┐
        │       Prediction Head               │
        │   [B, n_vars, pred_len]             │
        └─────────────────────────────────────┘
```

## 🔍 **關鍵實現細節回答**

### **1. 是否在 Embedding 前還是後？**
**答案：在 Embedding 內部！**

具體來說：
- 每個尺度**先獨立做 embedding**
- 然後在 embedding 的最後階段做 **fusion**
- Fusion 的結果才送入 Transformer Encoder

### **2. 是否本身就是 embedding？**
**答案：Fusion 是 Embedding 的一部分！**

我們的 `MultiScaleEnEmbedding` 類包含：
1. **Multi-scale patching**
2. **Per-scale linear embedding**
3. **Position embedding**
4. **Scale-specific global tokens**
5. **🔥 Fusion module** ← 這是關鍵創新
6. **最終輸出 dropout**

### **3. 最終是否 concat 成大序列？**
**答案：是的！**

不管用哪種 fusion 方法，最終都會：
```python
# 所有 fusion 方法的共同點
final_tokens = fusion_result  # [B, n_vars, total_patches, d_model]

# 重塑為 encoder 輸入格式
encoder_input = final_tokens.reshape(B * n_vars, total_patches, d_model)
```

這個 `total_patches = 25` 的序列就是送入 Transformer 的最終 token 序列。

### **4. 與原始架構的差異**

| 原始 TimeXer | 我們的 Multi-Scale TimeXer |
|-------------|---------------------------|
| 單一 patch size (16) | 多個 patch sizes [8,16,24] |
| 6 patches + 1 global = 7 tokens | 22 patches + 3 globals = 25 tokens |
| 簡單 linear embedding | 複雜 fusion embedding |
| 固定感受野 | 多尺度感受野 |
