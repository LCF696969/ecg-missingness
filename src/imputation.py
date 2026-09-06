"""M4 — 補值模組。

介面：impute(signal, mask, method) -> filled_signal

三種方法（plan_v3.md）：
  zero   零填補     — 最差基線，但它是判斷「排序有沒有翻轉」的關鍵參照
  ffill  前值填補   — 長缺口會變成水平線
  linear 線性內插   — 長缺口會被抹平，所有振盪消失

**零填補不是湊數。** 如果在超長連續缺口下，精緻補值跟填零表現差不多，
那本身就是有力的結論：長缺口的資訊補不回來，該做的是標記為不可用而不是硬補。

洩漏防護：這三種方法都是**逐筆、逐通道獨立**的，不使用任何跨樣本統計量，
因此不存在「用測試集統計量補值」的洩漏。若日後加入 KNN 或均值填補，
統計量必須只從訓練集計算。
"""
from __future__ import annotations

import numpy as np

METHODS = ("zero", "ffill", "linear")


def _impute_1ch(x: np.ndarray, m: np.ndarray, method: str) -> np.ndarray:
    """單一通道補值。x: (T,) float, m: (T,) bool, True = 缺失。"""
    out = x.copy()
    if not m.any():
        return out
    if m.all():
        # 整條通道皆缺：三種方法都無資訊可用，一律填 0 並在上層記錄
        out[:] = 0.0
        return out

    obs = ~m
    idx = np.arange(len(x))

    if method == "zero":
        out[m] = 0.0

    elif method == "ffill":
        # 前值填補：向前帶最後一個觀測值
        last = np.where(obs, idx, -1)
        np.maximum.accumulate(last, out=last)
        lead = last < 0                      # 開頭就缺，沒有前值可用
        out[~lead & m] = x[last[~lead & m]]
        # 開頭的缺口沒有前值，改用第一個觀測值回填（否則只能留 NaN）
        if lead.any():
            out[lead] = x[obs][0]

    elif method == "linear":
        # 線性內插；兩端超出觀測範圍的部分，np.interp 會自動以端點值延伸
        out[m] = np.interp(idx[m], idx[obs], x[obs])

    else:
        raise ValueError(f"未知的補值方法：{method}，可用 {METHODS}")

    return out


def impute(signal: np.ndarray, mask: np.ndarray, method: str) -> np.ndarray:
    """對一筆記錄補值。signal: (T, C)，mask: (T, C) bool。"""
    x = np.asarray(signal, dtype=np.float32)
    out = np.empty_like(x)
    for c in range(x.shape[1]):
        out[:, c] = _impute_1ch(x[:, c].astype(np.float64), mask[:, c], method)
    return out


def impute_batch(X: np.ndarray, M: np.ndarray, method: str) -> np.ndarray:
    """對一批記錄補值。X: (N, T, C)，M: (N, T, C) bool。"""
    out = np.empty_like(X, dtype=np.float32)
    for i in range(len(X)):
        out[i] = impute(X[i], M[i], method)
    return out


def reconstruction_error(X_true: np.ndarray, X_filled: np.ndarray,
                         M: np.ndarray) -> dict:
    """只在**被遮掉的點**上計算重建誤差。

    這是訊號層的評估，跟下游 AUROC 是兩件事——
    補得像不代表模型判讀得對，這兩者的落差本身就是本專案的看點之一。
    """
    d = (X_filled - X_true)[M]
    return {
        "mae": float(np.abs(d).mean()),
        "rmse": float(np.sqrt((d ** 2).mean())),
        "n_points": int(M.sum()),
    }
