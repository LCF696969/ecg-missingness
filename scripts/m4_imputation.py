"""M4 — 補值模組的驗證與視覺化。

檢查：
  1. 正確性：補值後無 NaN，且**觀測點必須完全沒有被改動**（補值不該碰到好資料）
  2. 邊界情況：開頭就缺、結尾就缺、整條通道皆缺
  3. 三線疊圖：原始 / 遮罩 / 補值
  4. 訊號層重建誤差：method × pattern × rate，看排序在什麼條件下翻轉

第 4 項是 plan 沒要求的加碼。它在**還沒進到 AUROC 之前**就先回答
「線性內插是不是永遠最好」，如果訊號層就已經翻轉，M5 的圖二就有故事。

產出：
  results/m4_correctness.csv
  results/m4_reconstruction_error.csv
  figures/m4_overlay.png
  figures/m4_reconstruction_error.png
  results/m4_log.txt
"""
from __future__ import annotations

import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config  # noqa: E402
from src import data as D  # noqa: E402
from src import imputation as IMP  # noqa: E402
from src import missingness as MS  # noqa: E402

RATES = [0.025, 0.05, 0.10, 0.20, 0.40]
PATTERN_KW = {"scatter": {}, "burst": {"k": 2}, "mnar": {}}


class Tee:
    def __init__(self, path: Path):
        self.file = open(path, "w", encoding="utf-8")

    def write(self, s):
        sys.__stdout__.write(s)
        self.file.write(s)
        self.file.flush()

    def flush(self):
        sys.__stdout__.flush()
        self.file.flush()

    def close(self):
        self.file.close()


def main() -> int:
    t0 = time.time()
    print("=" * 74)
    print(f"M4 — 補值模組驗證   {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 74)

    X, meta = D.load_full_cache()
    Xte = X[(meta.split == "test").values]
    del X
    sub = Xte[:300]
    T, C = Xte.shape[1], Xte.shape[2]
    print(f"\n[0] 測試集 {Xte.shape}，驗證子集 {sub.shape}")

    # ---------- 檢查 1：正確性 ----------
    print("\n[1] 正確性：補值後無 NaN，且觀測點未被改動")
    rows = []
    for pattern, kw in PATTERN_KW.items():
        M = MS.make_mask_batch(sub, pattern, 0.20, seed=0, **kw)
        for method in IMP.METHODS:
            F = IMP.impute_batch(sub, M, method)
            n_nan = int(np.isnan(F).sum())
            obs_changed = int((F[~M] != sub[~M]).sum())
            rows.append({"pattern": pattern, "method": method,
                         "n_nan": n_nan, "observed_points_changed": obs_changed,
                         "pass": (n_nan == 0 and obs_changed == 0)})
            print(f"    {pattern:8s} {method:7s}: NaN={n_nan}  "
                  f"觀測點被改動={obs_changed}  {'OK' if n_nan == 0 and obs_changed == 0 else '**失敗**'}")
    corr = pd.DataFrame(rows)
    corr.to_csv(config.RESULTS_DIR / "m4_correctness.csv", index=False)
    print(f"    {'全部通過' if corr['pass'].all() else '**有項目未通過**'}")

    # ---------- 檢查 2：邊界情況 ----------
    print("\n[2] 邊界情況")
    x = sub[0, :, 1].astype(np.float64)
    cases = {
        "開頭 50 點缺": np.r_[np.ones(50, bool), np.zeros(T - 50, bool)],
        "結尾 50 點缺": np.r_[np.zeros(T - 50, bool), np.ones(50, bool)],
        "整條通道皆缺": np.ones(T, bool),
    }
    for name, m in cases.items():
        line = f"    {name:14s}"
        for method in IMP.METHODS:
            f = IMP._impute_1ch(x, m, method)
            line += f"  {method}: NaN={int(np.isnan(f).sum())}"
        print(line)

    # ---------- 檢查 4：重建誤差 ----------
    print("\n[3] 訊號層重建誤差（只在被遮掉的點上計算，單位 mV）")
    rows = []
    for pattern, kw in PATTERN_KW.items():
        for rate in RATES:
            M = MS.make_mask_batch(sub, pattern, rate, seed=0, **kw)
            for method in IMP.METHODS:
                F = IMP.impute_batch(sub, M, method)
                e = IMP.reconstruction_error(sub, F, M)
                rows.append({"pattern": pattern, "rate": rate, "method": method,
                             "mae": e["mae"], "rmse": e["rmse"]})
    err = pd.DataFrame(rows)
    err.to_csv(config.RESULTS_DIR / "m4_reconstruction_error.csv", index=False)

    piv = err.pivot_table(index=["pattern", "rate"], columns="method", values="rmse")
    piv = piv[["zero", "ffill", "linear"]]
    piv["best"] = piv.idxmin(axis=1)
    print(piv.round(4).to_string())

    # burst 的 k 掃描（碎裂程度 vs 重建誤差）
    print("\n[4] 碎裂程度對重建誤差的影響（burst、缺失率 20%）")
    rows = []
    for k in [1, 2, 5, 10, 20]:
        M = MS.make_mask_batch(sub, "burst", 0.20, seed=0, k=k)
        for method in IMP.METHODS:
            F = IMP.impute_batch(sub, M, method)
            e = IMP.reconstruction_error(sub, F, M)
            rows.append({"k": k, "method": method, "rmse": e["rmse"]})
    kerr = pd.DataFrame(rows).pivot(index="k", columns="method", values="rmse")
    kerr = kerr[["zero", "ffill", "linear"]]
    kerr["best"] = kerr.idxmin(axis=1)
    print(kerr.round(4).to_string())
    kerr.to_csv(config.RESULTS_DIR / "m4_k_reconstruction_error.csv")

    # ---------- 檢查 3：三線疊圖 ----------
    print("\n[5] 三線疊圖")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    idx, lead, rate = 0, 1, 0.20
    t = np.arange(T) / config.SAMPLING_RATE
    x0 = Xte[idx, :, lead]
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True, sharey=True)
    rng = np.random.default_rng(0)
    m = MS.make_mask(T, C, "burst", rate, rng, k=2)
    for ax, method in zip(axes, IMP.METHODS):
        f = IMP.impute(Xte[idx], m, method)[:, lead]
        ax.plot(t, x0, lw=1.6, color="#b0b0b0", label="original (ground truth)", zorder=1)
        xm = x0.copy().astype(float)
        xm[m[:, lead]] = np.nan
        ax.plot(t, xm, lw=0.9, color="#1f4e79", label="observed", zorder=3)
        ax.plot(t, f, lw=1.0, color="#d62728", ls="--", label="imputed", zorder=2)
        d = np.diff(np.concatenate([[0], m[:, lead].astype(np.int8), [0]]))
        for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
            ax.axvspan(t[s], t[min(e, T - 1)], color="#d62728", alpha=0.10, lw=0, zorder=0)
        mae_val = np.abs(f[m[:, lead]] - x0[m[:, lead]]).mean()
        ax.set_title(f"{method}   —   MAE on missing points = {mae_val:.4f} mV",
                     fontsize=10, loc="left")
        ax.set_ylabel("mV")
        ax.grid(alpha=0.25, lw=0.4)
        ax.legend(fontsize=8, loc="upper right", ncol=3)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle("Imputation on burst missingness (k=2, 20% missing) — "
                 f"PTB-XL lead {config.LEAD_NAMES[lead]}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p = config.FIGURES_DIR / "m4_overlay.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print(f"    已存檔：{p}")

    # 重建誤差圖
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = {"zero": "#7f7f7f", "ffill": "#ff7f0e", "linear": "#1f77b4"}
    for method in IMP.METHODS:
        s = err[(err.pattern == "burst") & (err.method == method)]
        axes[0].plot(s.rate * 100, s.rmse, "o-", color=colors[method], label=method)
        axes[1].plot(kerr.index, kerr[method], "o-", color=colors[method], label=method)
    axes[0].set_xlabel("missing rate (%)"); axes[0].set_ylabel("RMSE on missing points (mV)")
    axes[0].set_title("Burst (k=2): reconstruction error vs missing rate", fontsize=10)
    axes[1].set_xlabel("k (number of gaps, 20% missing)"); axes[1].set_ylabel("RMSE (mV)")
    axes[1].set_title("Fragmentation: fewer, longer gaps → harder to reconstruct", fontsize=10)
    axes[1].set_xscale("log"); axes[1].set_xticks(kerr.index)
    axes[1].set_xticklabels(kerr.index)
    for ax in axes:
        ax.grid(alpha=0.3, lw=0.4); ax.legend(fontsize=9)
    fig.tight_layout()
    p2 = config.FIGURES_DIR / "m4_reconstruction_error.png"
    fig.savefig(p2, dpi=140)
    plt.close(fig)
    print(f"    已存檔：{p2}")

    print("\n" + "=" * 74)
    print(f"M4 完成，總耗時 {time.time() - t0:.1f}s")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    log_path = config.RESULTS_DIR / "m4_log.txt"
    tee = Tee(log_path)
    sys.stdout = tee
    try:
        code = main()
    except Exception:
        print("\n**執行失敗**\n")
        traceback.print_exc(file=tee)
        code = 1
    finally:
        sys.stdout = sys.__stdout__
        tee.close()
    print(f"log 已寫入：{log_path}")
    sys.exit(code)
