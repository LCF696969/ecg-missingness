"""M6 — 失效個案圖：三種補值失敗的機制，各挑一個真實個案。

主圖與門檻表回答的是「平均而言傷害多大」。這張圖回答的是**「它是怎麼壞的」**——
把訊號層發生的事、和模型判讀翻轉這兩件事放在同一張圖上，讓機制可以被看見而不是被宣稱。

三個個案分別對應三種機制，而且**兩個方向的錯誤都涵蓋**：
  A  前值填補 × 2 秒連續缺口  → 缺口被拉成水平線，異常波形被抹平 → **漏診**（把異常讀成正常）
  B  零填補   × 20% 散布缺失  → 每個缺點製造一次跳到 0 的人工不連續，高頻功率暴增 → 誤判為異常
  C  線性內插 × 20% 訊號相關  → 高振幅的 QRS 被優先吃掉，內插畫出平滑斜坡 → 誤判為異常

每個 panel 都附上一個**可量測的機制指標**（最長平坦段 / 15–40Hz 相對功率 / 峰對峰振幅），
所以「為什麼會壞」是有數字支撐的，不是看圖說故事。

產出：
  figures/m6_failure_cases.png
  results/m6_failure_cases.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import joblib  # noqa: E402
from scipy import signal as sps  # noqa: E402

import config  # noqa: E402
from src import features as F, imputation as IMP, missingness as MS  # noqa: E402

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8a85", "#e3e3df"
SURFACE = "#fcfcfb"
TRUE_C = "#8a8a85"                      # 真實訊號：中性灰，是參照不是主角
IMP_C = {"zero": "#4a3aa7", "ffill": "#eda100", "linear": "#e34948"}
GAP_C = "#e34948"                       # 缺失區間的底色（極淡）
FS, T = 100, 1000

CASES = [
    dict(tag="A", ecg_id=4785,  idx=593,  pattern="burst",   rate=0.20, k=1,
         imputer="ffill",
         title="Forward-fill turns a 2 s gap into a flat line",
         mech="flat", window=(0.0, 10.0), note_dy=-58, ylim_q=(0.0, 100.0),
         note="the abnormality is erased"),
    dict(tag="B", ecg_id=473,   idx=61,   pattern="scatter", rate=0.20, k=None,
         imputer="zero",
         title="Zero-fill on scattered gaps injects high-frequency steps",
         mech="hf", window=(2.0, 4.0), note_dy=-52, ylim_q=(0.2, 99.8),
         note="every gap is a jump to 0 and back"),
    dict(tag="C", ecg_id=13025, idx=1406, pattern="mnar",    rate=0.20, k=None,
         imputer="linear",
         title="Signal-dependent loss eats the QRS; interpolation draws a smooth ramp",
         mech="p2p", window=(0.0, 10.0), note_dy=-54, ylim_q=(0.0, 100.0),
         note="the beat is replaced by a straight line"),
]


def load_test():
    info = json.loads((config.CACHE_DIR / "full_info.json").read_text(encoding="utf-8"))
    meta = pd.read_csv(config.CACHE_DIR / "full_meta.csv", index_col="ecg_id")
    pos = np.where(meta["split"].values == "test")[0]
    parts = []
    for c in info["chunks"]:
        arr = np.load(config.CACHE_DIR / c["file"], mmap_mode="r")
        sel = pos[(pos >= c["start"]) & (pos < c["end"])] - c["start"]
        if len(sel):
            parts.append(np.asarray(arr[sel]))
    X = np.concatenate(parts, 0).astype(np.float32) * np.float32(info["scale_to_mV"])
    return X, meta["label"].values[pos], meta.index.values[pos]


def longest_flat_seconds(x: np.ndarray, eps: float = 1e-9) -> float:
    """最長的「連續不變」區段長度（秒）。前值填補的直接指紋。"""
    flat = np.abs(np.diff(x)) <= eps
    best = run = 0
    for f in flat:
        run = run + 1 if f else 0
        best = max(best, run)
    return best / FS


def relpow_15_40(x: np.ndarray) -> float:
    f, p = sps.welch(x, fs=FS, nperseg=min(256, len(x)))
    tot = p[(f >= 0.5) & (f <= 40)].sum()
    return float(p[(f >= 15) & (f <= 40)].sum() / tot) if tot > 0 else float("nan")


def style(ax):
    ax.grid(True, axis="y", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8.5, length=0)


def main() -> int:
    X, y, ecg_ids = load_test()
    model = joblib.load(PROJECT_ROOT / "models" / "m2_full_seed0.joblib")

    fig, axes = plt.subplots(3, 1, figsize=(11.2, 10.4))
    fig.patch.set_facecolor(SURFACE)
    rows = []

    for ax, cs in zip(axes, CASES):
        style(ax)
        i, imp = cs["idx"], cs["imputer"]
        assert int(ecg_ids[i]) == cs["ecg_id"], "index / ecg_id 不一致，快取變了？"

        kw = {"k": cs["k"]} if cs["pattern"] == "burst" else {}
        M = MS.make_mask_batch(X, cs["pattern"], cs["rate"], 0, **kw)
        Xi = IMP.impute_batch(X, M, imp)

        p_clean = float(model.predict_proba(F.extract_features(X[i][None]))[0, 1])
        p_dirty = float(model.predict_proba(F.extract_features(Xi[i][None]))[0, 1])

        # 挑「補值誤差最大的那個導程」來畫——那是模型實際看到最嚴重的破壞。
        # 用 RMS 而不是 max：max 會被單一假影尖峰綁架，選出一個不具代表性的導程
        # （第一版就是這樣選到 V1 開頭一根 8 mV 的假影，整條線被壓扁看不見機制）。
        err = np.sqrt(((Xi[i] - X[i]) ** 2).mean(axis=0))
        lead = int(np.argmax(err))
        xt_full, xi_full, m_full = X[i, :, lead], Xi[i, :, lead], M[i, :, lead]
        w0, w1 = cs["window"]
        a, b = int(w0 * FS), int(w1 * FS)
        xt, xi, m = xt_full[a:b], xi_full[a:b], m_full[a:b]
        t = np.arange(a, b) / FS

        # 缺失區間底色（是註記，不是資料序列，所以不進圖例）
        edges = np.diff(np.concatenate([[0], m.astype(int), [0]]))
        for s, e in zip(np.where(edges == 1)[0], np.where(edges == -1)[0]):
            ax.axvspan(s / FS, e / FS, color=GAP_C, alpha=0.10, lw=0, zorder=1)

        ax.plot(t, xt, color=TRUE_C, lw=1.0, zorder=2,
                label="original signal" if cs["tag"] == "A" else None)
        ax.plot(t, xi, color=IMP_C[imp], lw=1.7, zorder=3,
                label="after imputation" if cs["tag"] == "A" else None)

        if cs["mech"] == "flat":
            mv, mvd = longest_flat_seconds(xt_full), longest_flat_seconds(xi_full)
            metric = f"longest flat run  {mv:.2f} s → {mvd:.2f} s"
        elif cs["mech"] == "hf":
            mv, mvd = relpow_15_40(xt_full), relpow_15_40(xi_full)
            metric = f"15–40 Hz relative power  {mv:.3f} → {mvd:.3f}"
        else:
            mv, mvd = float(xt_full.max() - xt_full.min()), float(xi_full.max() - xi_full.min())
            metric = f"peak-to-peak amplitude  {mv:.2f} mV → {mvd:.2f} mV"

        truth = "ABNORMAL" if y[i] == 1 else "NORMAL"
        said_c = "abnormal" if p_clean > 0.5 else "normal"
        said_d = "abnormal" if p_dirty > 0.5 else "normal"
        conf_c = p_clean if p_clean > 0.5 else 1 - p_clean
        conf_d = p_dirty if p_dirty > 0.5 else 1 - p_dirty
        kind = "MISSED a real abnormality" if y[i] == 1 else "FALSE ALARM on a normal ECG"

        ax.set_title(f"{cs['tag']} · {cs['title']}", color=INK, fontsize=12,
                     loc="left", pad=30)
        ax.text(0, 1.115,
                f"ecg_id {cs['ecg_id']} · lead {config.LEAD_NAMES[lead]} · truth: {truth}"
                f"   —   {cs['pattern']} {cs['rate']:.0%}"
                + (f" (k={cs['k']})" if cs["k"] else "") + f" · {imp}"
                + ("" if (w1 - w0) >= 10 else f"   ·   zoomed to {w0:.0f}–{w1:.0f} s so the individual steps are visible"),
                transform=ax.transAxes, color=INK2, fontsize=8.8)
        ax.text(0, 1.035,
                f"model said {said_c} ({conf_c:.1%})  →  {said_d} ({conf_d:.1%})"
                f"     ✗ {kind}     ·     {metric}",
                transform=ax.transAxes, color=INK, fontsize=9.3, fontweight="bold")

        ax.set_ylabel("mV", color=INK2, fontsize=9)
        ax.set_xlim(w0, w1)
        # y 範圍：預設用完整 min/max，因為 panel C 要證明的正是「原始波峰被抹掉」,
        # 裁掉波峰等於把證據裁掉。只有 panel B 那筆帶假影尖峰的記錄才收緊分位數。
        both = np.concatenate([xt, xi])
        lo_q, hi_q = np.percentile(both, list(cs["ylim_q"]))
        pad = 0.16 * (hi_q - lo_q)
        ax.set_ylim(lo_q - pad, hi_q + pad * 1.4)
        if cs["tag"] == "C":
            ax.set_xlabel("time (s)", color=INK2, fontsize=9.5)

        # 指向最大缺口，把機制指出來
        win_edges = np.diff(np.concatenate([[0], m.astype(int), [0]]))
        widths = [(e - s, s, e) for s, e in
                  zip(np.where(win_edges == 1)[0], np.where(win_edges == -1)[0])]
        if widths:
            _, s, e = max(widths)
            xm = (a + (s + e) / 2) / FS
            # 短的直引線，就標在目標旁邊。第一版把文字丟到角落再拉一條長弧線過去，
            # 那條線橫掃整張圖還壓過波形——註記不該比資料還顯眼。
            ax.annotate(cs["note"], xy=(xm, float(xi[(s + e) // 2])),
                        xytext=(0, cs["note_dy"]), textcoords="offset points",
                        ha="center", va="center", color=INK, fontsize=8.8,
                        bbox=dict(boxstyle="round,pad=0.32", fc=SURFACE,
                                  ec=GRID, lw=0.8, alpha=0.94),
                        arrowprops=dict(arrowstyle="->", color=INK2, lw=1.0,
                                        shrinkA=3, shrinkB=4))

        rows.append(dict(tag=cs["tag"], ecg_id=cs["ecg_id"], lead=config.LEAD_NAMES[lead],
                         truth=truth, pattern=cs["pattern"], rate=cs["rate"], k=cs["k"],
                         imputer=imp, p_clean=round(p_clean, 4), p_dirty=round(p_dirty, 4),
                         error_kind="false negative" if y[i] == 1 else "false positive",
                         metric=cs["mech"], metric_clean=round(mv, 4),
                         metric_imputed=round(mvd, 4)))

    axes[0].legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper right", ncol=2)
    fig.suptitle("How the model actually fails — three mechanisms, three real records",
                 color=INK, fontsize=14, x=0.008, ha="left", y=0.995)
    fig.text(0.008, 0.965,
             "Shaded bands are the missing samples. Same model, same record — only the "
             "imputed values differ. Both error directions appear.",
             color=INK2, fontsize=9.2, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.945])
    out = config.FIGURES_DIR / "m6_failure_cases.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)

    df = pd.DataFrame(rows)
    df.to_csv(config.RESULTS_DIR / "m6_failure_cases.csv", index=False)
    print(df.to_string(index=False))
    print("\n已輸出：", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
