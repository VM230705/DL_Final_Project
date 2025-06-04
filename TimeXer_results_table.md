# Long-term Forecasting Results - TimeXer Model

## Results Summary Table

| Dataset | Prediction Length | MSE | MAE | Model Configuration | check |
|---------|------------------|-----|-----|-------------------|------|
| **ETTh1** | 96 | 0.3818 | 0.4029 | dm256_nh8_el1_dl1_df2048 |[]|
| ETTh1 | 192 | 0.4285 | 0.4355 | dm128_nh8_el2_dl1_df2048 |[]|
| ETTh1 | 336 | 0.4677 | 0.4484 | dm512_nh8_el1_dl1_df1024 |[]|
| ETTh1 | 720 | 0.4694 | 0.4612 | dm256_nh8_el1_dl1_df1024 |[]|
| **ETTh2** | 96 | 0.2857 | 0.3377 | dm256_nh8_el1_dl1_df1024 |[]|
| ETTh2 | 192 | 0.3629 | 0.3892 | dm256_nh8_el1_dl1_df1024 |[]|
| ETTh2 | 336 | 0.4136 | 0.4229 | dm512_nh8_el2_dl1_df1024 |[]|
| ETTh2 | 720 | 0.4076 | 0.4322 | dm256_nh8_el2_dl1_df1024 |[]|
| **ETTm1** | 96 | 0.3178 | 0.3563 | dm256_nh8_el1_dl1_df2048 |[]|
| ETTm1 | 192 | 0.3616 | 0.3829 | dm256_nh8_el1_dl1_df256 |[]|
| ETTm1 | 336 | 0.3951 | 0.4067 | dm256_nh8_el1_dl1_df1024 |[]|
| ETTm1 | 720 | 0.4524 | 0.4413 | dm256_nh8_el1_dl1_df512 |[]|
| **ETTm2** | 96 | 0.1708 | 0.2557 | dm256_nh8_el1_dl1_df2048 |[]|
| ETTm2 | 192 | 0.2368 | 0.2990 | dm256_nh8_el1_dl1_df1024 |[]|
| ETTm2 | 336 | 0.2959 | 0.3382 | dm512_nh8_el1_dl1_df1024 |[]|
| ETTm2 | 720 | 0.3920 | 0.3937 | dm512_nh8_el1_dl1_df2048 |[]|
| **Exchange** | 96 | 0.0885 | 0.2067 | dm512_nh8_el2_dl1_df2048 |[]|
| Exchange | 192 | 0.1826 | 0.3031 | dm512_nh8_el2_dl1_df2048 |[]|
| Exchange | 336 | 0.3481 | 0.4252 | dm512_nh8_el2_dl1_df2048 |[]|
| Exchange | 720 | 0.8811 | 0.7045 | dm512_nh8_el2_dl1_df2048 |[]|
| **Weather** | 96 | 0.1574 | 0.2047 | dm256_nh8_el1_dl1_df512 |[x]|
| Weather | 192 | 0.2041 | 0.2475 | dm128_nh8_el3_dl1_df1024 |[]|
| Weather | 336 | 0.2607 | 0.2897 | dm256_nh8_el1_dl1_df2048 |[]|
| Weather | 720 | 0.3403 | 0.3408 | dm128_nh8_el1_dl1_df2048 |[]|
| **ECL** | 96 | 0.1404 | 0.2425 | dm512_nh8_el4_dl1_df512 |[]|
| ECL | 192 | 0.1575 | 0.2559 | dm512_nh8_el3_dl1_df2048 |[]|
| ECL | 336 | 0.1764 | 0.2751 | dm512_nh8_el4_dl1_df2048 |[]|
| ECL | 720 | 0.2106 | 0.3060 | dm512_nh8_el3_dl1_df2048 |[]|
| **Traffic** | 96 | 0.4281 | 0.2707 | dm512_nh8_el3_dl1_df512 |[]|
| Traffic | 192 | 0.4482 | 0.2817 | dm512_nh8_el3_dl1_df512 |[]|
| Traffic | 336 | 0.4716 | 0.2890 | dm512_nh8_el2_dl1_df512 |[]|
| Traffic | 720 | 0.5137 | 0.3069 | dm512_nh8_el2_dl1_df512 |[]|

## Performance Summary by Dataset

### Best MSE Performance
1. **Exchange (96)**: 0.0885
2. **ECL (96)**: 0.1404
3. **Weather (96)**: 0.1574
4. **ETTm2 (96)**: 0.1708
5. **ETTh2 (96)**: 0.2857

### Best MAE Performance
1. **Exchange (96)**: 0.2067
2. **Weather (96)**: 0.2047
3. **ECL (96)**: 0.2425
4. **ETTm2 (96)**: 0.2557
5. **Traffic (96)**: 0.2707

### Observations
- **Exchange** dataset shows the best performance for short-term predictions (96) but degrades significantly for long-term predictions (720)
- **ECL** and **Weather** datasets maintain relatively stable performance across different prediction lengths
- **Traffic** dataset shows consistent performance with moderate degradation as prediction length increases
- All datasets show performance degradation as prediction length increases from 96 to 720 time steps