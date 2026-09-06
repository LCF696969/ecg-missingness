"""M3 — 缺失注入器（本專案技術核心）。

統一介面：
    make_mask(T, C, pattern, rate, rng, signal=None, **kw) -> (T, C) bool, True = 缺失

三種型態（用語精確度見 notes/process_and_decisions.md 的 C1）：
  scatter — 每個時間點獨立遺失。Rubin 分類：MCAR。
  burst   — 缺失集中成 k 段連續缺口。Rubin 分類：**仍是 MCAR**（缺口位置與訊號值無關），
            改變的是缺失的「結構」而非「機制」。
  mnar    — 缺失機率隨訊號振幅上升，被遮掉的正是那段訊號本身。Rubin 分類：MNAR。

設計決定：**各通道獨立生成遮罩**（不是 12 導程同步消失）。
本研究關心的是單一通道內缺失的時間結構；多通道同步失效成因不同（電極脫落、線路故障），
屬於另一種失效模式，刻意排除在範圍外——理由見 notes/process_and_decisions.md 的 D9。
若要模擬同步缺失，把 per_channel=False 打開。
"""
from __future__ import annotations

import numpy as np

PATTERNS = ("scatter", "burst", "mnar")


# ---------------------------------------------------------------- scatter
def _mask_scatter_1ch(T: int, n_missing: int, rng: np.random.Generator) -> np.ndarray:
    m = np.zeros(T, dtype=bool)
    if n_missing > 0:
        m[rng.choice(T, size=n_missing, replace=False)] = True
    return m


# ------------------------------------------------------------------ burst
def _mask_burst_1ch(T: int, n_missing: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """把 n_missing 個缺失點切成 k 段連續缺口，隨機不重疊擺放。

    段長盡量平均：base = n_missing // k，餘數 r 段各多 1 點，
    確保**總缺失數精確等於 n_missing**（缺失率驗證的關鍵）。
    """
    m = np.zeros(T, dtype=bool)
    if n_missing <= 0:
        return m
    k = max(1, min(k, n_missing))          # 段數不能多於缺失點數
    base, r = divmod(n_missing, k)
    lengths = np.array([base + 1] * r + [base] * (k - r))
    rng.shuffle(lengths)

    free = T - n_missing                    # 可用來當間隔的空白點數
    if free < 0:
        return np.ones(T, dtype=bool)
    # 在 free+1 個可能的插入位置中選 k 個（可重複），排序後依序展開，保證不重疊
    cuts = np.sort(rng.integers(0, free + 1, size=k))
    offsets = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    starts = cuts + offsets
    for s, L in zip(starts, lengths):
        m[s:s + L] = True
    return m


# ------------------------------------------------------------------- MNAR
def _mask_mnar_1ch(x: np.ndarray, n_missing: int, rng: np.random.Generator,
                   window: int = 20, alpha: float = 2.0) -> np.ndarray:
    """訊號相關缺失：視窗振幅越大越容易被遮掉。

    心電圖振幅最大的地方就是 QRS，所以這個型態會**優先吃掉診斷資訊最集中的部分**，
    模擬病人躁動時電極接觸不良（動得越厲害、訊號越大、也越容易掉）。

    為了讓缺失率精確，最後一個視窗只遮部分點（遮該視窗中振幅最大的那幾點）。
    """
    T = len(x)
    m = np.zeros(T, dtype=bool)
    if n_missing <= 0:
        return m

    n_win = T // window
    if n_win == 0:
        return _mask_scatter_1ch(T, n_missing, rng)

    xw = x[:n_win * window].reshape(n_win, window)
    amp = xw.max(axis=1) - xw.min(axis=1)          # 視窗內振幅（peak-to-peak）
    w = np.power(amp - amp.min() + 1e-9, alpha)
    w = w / w.sum()

    n_full, rem = divmod(n_missing, window)
    n_pick = min(n_full + (1 if rem else 0), n_win)
    chosen = rng.choice(n_win, size=n_pick, replace=False, p=w)

    full_part = chosen[:n_full] if n_full <= len(chosen) else chosen
    for wi in full_part:
        m[wi * window:(wi + 1) * window] = True

    if rem and len(chosen) > n_full:
        wi = chosen[n_full]
        seg = np.abs(xw[wi] - xw[wi].mean())
        top = np.argsort(seg)[-rem:]               # 該視窗中偏離最大的 rem 個點
        m[wi * window + top] = True

    # 若因視窗數不足而遮得不夠，補隨機點（極端高缺失率時才會發生）
    short = n_missing - int(m.sum())
    if short > 0:
        avail = np.flatnonzero(~m)
        m[rng.choice(avail, size=min(short, len(avail)), replace=False)] = True
    return m


# --------------------------------------------------------------- 統一介面
def make_mask(T: int, C: int, pattern: str, rate: float,
              rng: np.random.Generator, signal: np.ndarray | None = None,
              k: int = 3, window: int = 20, alpha: float = 2.0,
              per_channel: bool = True) -> np.ndarray:
    """回傳 (T, C) 布林陣列，True = 缺失。

    參數
    ----
    pattern   : "scatter" | "burst" | "mnar"
    rate      : 目標缺失比例（0–1）
    signal    : (T, C)，mnar 必須提供
    k         : burst 的缺口段數
    window    : mnar 的視窗長度（取樣點）
    alpha     : mnar 的振幅權重指數；0 = 退化成隨機選視窗
    per_channel : True 時每個導程獨立生成遮罩；False 時所有導程共用同一個遮罩
    """
    if pattern not in PATTERNS:
        raise ValueError(f"未知的 pattern：{pattern}，可用 {PATTERNS}")
    if pattern == "mnar" and signal is None:
        raise ValueError("mnar 型態需要提供 signal")

    n_missing = int(round(rate * T))
    n_gen = C if per_channel else 1

    cols = []
    for c in range(n_gen):
        if pattern == "scatter":
            cols.append(_mask_scatter_1ch(T, n_missing, rng))
        elif pattern == "burst":
            cols.append(_mask_burst_1ch(T, n_missing, k, rng))
        else:
            x = signal[:, c] if per_channel else signal.mean(axis=1)
            cols.append(_mask_mnar_1ch(np.asarray(x, dtype=np.float64),
                                       n_missing, rng, window=window, alpha=alpha))

    M = np.stack(cols, axis=1)
    if not per_channel:
        M = np.repeat(M, C, axis=1)
    return M


def make_mask_batch(X: np.ndarray, pattern: str, rate: float, seed: int,
                    **kw) -> np.ndarray:
    """對一批訊號 (N, T, C) 生成遮罩，回傳同形狀的布林陣列。

    固定 seed 時結果完全可重現（M3 的三個必做檢查之一）。
    """
    n, T, C = X.shape
    rng = np.random.default_rng(seed)
    out = np.empty((n, T, C), dtype=bool)
    for i in range(n):
        out[i] = make_mask(T, C, pattern, rate, rng,
                           signal=X[i] if pattern == "mnar" else None, **kw)
    return out


# ------------------------------------------------------------- 診斷用工具
def gap_stats(mask_1ch: np.ndarray) -> dict:
    """量測單一通道遮罩的缺口結構：段數、各段長度。

    用來驗證 burst 實際切出幾段（相鄰缺口可能合併，導致實際段數少於設定的 k）。
    """
    m = mask_1ch.astype(np.int8)
    d = np.diff(np.concatenate([[0], m, [0]]))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    lengths = ends - starts
    return {
        "n_gaps": int(len(lengths)),
        "total_missing": int(m.sum()),
        "max_gap": int(lengths.max()) if len(lengths) else 0,
        "mean_gap": float(lengths.mean()) if len(lengths) else 0.0,
        "lengths": lengths,
    }
