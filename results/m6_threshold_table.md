# 資料品質門檻表

基準：無缺失時 AUROC = 0.8894（PTB-XL fold 10，2158 筆，固定模型）

「最大可接受缺失率」= 在該型態下，效能損失仍小於容忍上限的最高缺失率；
補值方法取該缺失率下傷害最小者。

## 容忍上限 ΔAUROC < 0.01

| 缺失型態 | 最大可接受比例 | 該點的補值方法 | 該點 AUROC | 高缺失率下建議的補值 |
|---|---|---|---|---|
| Scattered (MCAR) | 20.0% | ffill | 0.8872 | ffill |
| Burst (MCAR, structured) | 20.0% | zero | 0.8861 | zero |
| Signal-dependent (MNAR) | 10.0% | zero | 0.8849 | ffill |

## 容忍上限 ΔAUROC < 0.03

| 缺失型態 | 最大可接受比例 | 該點的補值方法 | 該點 AUROC | 高缺失率下建議的補值 |
|---|---|---|---|---|
| Scattered (MCAR) | 40.0% | ffill | 0.8789 | ffill |
| Burst (MCAR, structured) | 40.0% | zero | 0.8675 | zero |
| Signal-dependent (MNAR) | 10.0% | zero | 0.8849 | ffill |
