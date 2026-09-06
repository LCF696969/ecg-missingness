"""M2：特徵工程。

每個導程算 16 個特徵，12 導程串接 = 192 維。

設計原則（重要）：
- **所有特徵都是「逐筆、逐導程獨立」計算的**，沒有用到任何跨樣本的統計量
  （例如全體平均、標準化參數）。因此不存在訓練集統計量洩漏到測試集的問題，
  也不需要 StandardScaler（XGBoost 對單調變換不敏感）。
- Hjorth 三參數是核心：它描述訊號的「形狀複雜度」，對缺失與補值造成的
  波形破壞特別敏感——長缺口被線性內插抹平後，mobility 與 complexity 會明顯下降。
"""
from __future__ import annotations

import numpy as np
from scipy import signal as sps
from scipy import stats as spstats

# 每導程的特徵名稱（順序即為輸出順序）
FEATURE_NAMES = [
    "mean", "std", "min", "max", "rms",
    "skew", "kurtosis",
    "diff_std", "diff_absmean",
    "zero_cross_rate",
    "hjorth_activity", "hjorth_mobility", "hjorth_complexity",
    "relpow_0.5_4", "relpow_4_15", "relpow_15_40",
]
N_FEATURES_PER_LEAD = len(FEATURE_NAMES)

BANDS = [(0.5, 4.0), (4.0, 15.0), (15.0, 40.0)]
_EPS = 1e-12


def hjorth(x: np.ndarray, axis: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hjorth 參數：activity（變異數）、mobility、complexity。

    mobility   = sqrt(var(dx) / var(x))
    complexity = mobility(dx) / mobility(x)
    """
    dx = np.diff(x, axis=axis)
    ddx = np.diff(dx, axis=axis)

    var_x = np.var(x, axis=axis) + _EPS
    var_dx = np.var(dx, axis=axis) + _EPS
    var_ddx = np.var(ddx, axis=axis) + _EPS

    mob_x = np.sqrt(var_dx / var_x)
    mob_dx = np.sqrt(var_ddx / var_dx)
    complexity = mob_dx / (mob_x + _EPS)
    return var_x, mob_x, complexity


def band_relative_power(X: np.ndarray, fs: int = 100) -> np.ndarray:
    """各頻帶相對功率，回傳 (N, C, len(BANDS))。

    以 Welch 法估功率譜；分母是 0.5–49Hz 的總功率，避免直流分量主導。
    """
    n, t, c = X.shape
    # scipy 對最後一軸做 FFT，所以轉成 (N, C, T)
    Xt = np.transpose(X, (0, 2, 1))
    nperseg = min(256, t)
    freqs, psd = sps.welch(Xt, fs=fs, nperseg=nperseg, axis=-1)

    total_mask = (freqs >= 0.5) & (freqs <= 49.0)
    total = psd[..., total_mask].sum(axis=-1) + _EPS

    out = np.empty((n, c, len(BANDS)), dtype=np.float32)
    for i, (lo, hi) in enumerate(BANDS):
        m = (freqs >= lo) & (freqs < hi)
        out[..., i] = (psd[..., m].sum(axis=-1) / total).astype(np.float32)
    return out


def extract_features(X: np.ndarray, fs: int = 100, batch_size: int = 2000,
                     verbose: bool = False) -> np.ndarray:
    """輸入 (N, T, C)，輸出 (N, C * 16) 的特徵矩陣（float32）。

    **分批處理**：全量 21388 筆若一次轉成 float64 會佔用 2GB 以上，
    加上 Welch 的中間緩衝會超過記憶體，因此預設每 2000 筆處理一次。
    分批不影響結果——所有特徵都是逐筆獨立計算的。
    """
    n = len(X)
    if n <= batch_size:
        return _extract_batch(X, fs=fs)
    out = np.empty((n, X.shape[2] * N_FEATURES_PER_LEAD), dtype=np.float32)
    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        out[s:e] = _extract_batch(X[s:e], fs=fs)
        if verbose:
            print(f"    特徵抽取 {e}/{n} ...", flush=True)
    return out


def _extract_batch(X: np.ndarray, fs: int = 100) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    n, t, c = X.shape

    mean = X.mean(axis=1)
    std = X.std(axis=1)
    xmin = X.min(axis=1)
    xmax = X.max(axis=1)
    rms = np.sqrt((X ** 2).mean(axis=1))
    skew = spstats.skew(X, axis=1)
    kurt = spstats.kurtosis(X, axis=1)

    dX = np.diff(X, axis=1)
    diff_std = dX.std(axis=1)
    diff_absmean = np.abs(dX).mean(axis=1)

    # 過零率：相對於各導程自身均值的過零次數比例
    centered = X - mean[:, None, :]
    zc = (np.diff(np.signbit(centered), axis=1) != 0).sum(axis=1) / (t - 1)

    act, mob, comp = hjorth(X, axis=1)
    relpow = band_relative_power(X.astype(np.float32), fs=fs)  # (N, C, 3)

    per_lead = np.stack(
        [mean, std, xmin, xmax, rms, skew, kurt,
         diff_std, diff_absmean, zc, act, mob, comp],
        axis=-1,
    )                                            # (N, C, 13)
    per_lead = np.concatenate([per_lead, relpow], axis=-1)  # (N, C, 16)

    feats = per_lead.reshape(n, c * N_FEATURES_PER_LEAD).astype(np.float32)
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    return feats


def feature_column_names(lead_names: list[str]) -> list[str]:
    return [f"{lead}_{f}" for lead in lead_names for f in FEATURE_NAMES]
