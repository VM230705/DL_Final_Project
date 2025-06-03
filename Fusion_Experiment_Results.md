# 融合機制實驗結果紀錄

本文檔記錄了四個指定資料夾中所有融合方法和基線模型的實驗數據。

## 實驗設置
- **數據集**: Weather
- **輸入長度**: 96
- **預測長度**: 96, 192, 336, 720
- **模型**: TimeXer
- **評估指標**: MSE (Mean Squared Error), MAE (Mean Absolute Error)

## 實驗結果

### 1. 預測長度 96 (fusion_comparison_logs_20250602_210417)

| 方法 | MSE | MAE |
|------|-----|-----|
| Single Scale (Baseline) | 0.1574 | 0.2047 |
| Multi Attention | **0.1566** | **0.2043** |
| Multi Concat | 0.1580 | 0.2060 |
| Multi Gated | 0.1582 | 0.2061 |
| Multi Hierarchical | 0.1573 | 0.2054 |

**最佳結果**: Multi Attention 在 MSE 和 MAE 指標上都表現最佳

### 2. 預測長度 192 (fusion_comparison_logs_20250602_211241)

| 方法 | MSE | MAE |
|------|-----|-----|
| Single Scale (Baseline) | 0.2041 | 0.2475 |
| Multi Attention | 0.2059 | 0.2497 |
| Multi Concat | 0.2041 | 0.2483 |
| Multi Gated | 0.2043 | 0.2473 |
| Multi Hierarchical | **0.2037** | **0.2479** |

**最佳結果**: Multi Hierarchical 在 MSE 指標上表現最佳，Multi Gated 在 MAE 指標上表現最佳

### 3. 預測長度 336 (fusion_comparison_logs_20250602_211323)

| 方法 | MSE | MAE |
|------|-----|-----|
| Single Scale (Baseline) | **0.2607** | **0.2897** |
| Multi Attention | 0.2638 | 0.2918 |
| Multi Concat | 0.2634 | 0.2918 |
| Multi Gated | 0.2622 | 0.2908 |
| Multi Hierarchical | 0.2616 | 0.2911 |

**最佳結果**: Single Scale (Baseline) 在兩個指標上都表現最佳

### 4. 預測長度 720 (fusion_comparison_logs_20250602_211334)

| 方法 | MSE | MAE |
|------|-----|-----|
| Single Scale (Baseline) | **0.3403** | **0.3408** |
| Multi Attention | 0.3446 | 0.3453 |
| Multi Gated | 0.3431 | 0.3425 |
| Multi Hierarchical | - | - |

**注意**: Multi Hierarchical 和 Multi Concat 在預測長度 720 的實驗中沒有完成或結果不可用

**最佳結果**: Single Scale (Baseline) 在兩個指標上都表現最佳

## 分析總結

### 關鍵發現

1. **短期預測優勢**: 在較短的預測長度 (96) 下，Multi Attention 融合方法表現最佳
2. **中期預測**: 在預測長度 192 時，Multi Hierarchical 表現略勝一籌
3. **長期預測挑戰**: 在較長的預測長度 (336, 720) 下，傳統的 Single Scale 基線模型反而表現最佳
4. **融合方法的局限性**: 隨著預測長度增加，多尺度融合方法的優勢逐漸減弱

### 性能趨勢

- **MSE 趨勢**: 隨著預測長度增加，所有方法的 MSE 都呈上升趨勢
- **MAE 趨勢**: 同樣隨預測長度增加而增加，但增長率相對較緩

### 方法比較

1. **Multi Attention**: 在短期預測中表現優異，但在長期預測中性能下降明顯
2. **Multi Hierarchical**: 在中期預測中表現穩定
3. **Multi Gated**: 整體表現中等，在各個預測長度下都有一定競爭力
4. **Multi Concat**: 表現一般，沒有明顯優勢
5. **Single Scale (Baseline)**: 在長期預測中展現出良好的穩定性

## 建議

1. **根據預測長度選擇方法**: 短期預測推薦 Multi Attention，長期預測建議使用 Single Scale
2. **進一步優化**: 需要針對長期預測場景優化多尺度融合策略
3. **穩定性考慮**: Single Scale 方法在各種預測長度下都表現穩定，可作為可靠的基線