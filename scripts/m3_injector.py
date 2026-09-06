"""M3 — 缺失注入器的驗證與視覺化。

執行三個必做檢查（plan_v3.md）：
  1. 實際缺失率驗證：每種型態、每個缺失率，實測比例與目標誤差 < 1%
  2. 可重現性：同 seed 同輸入 → 完全相同的遮罩
  3. 三宮格視覺化：同一筆訊號、同樣缺失率、三種型態並排

外加兩項本專案自訂的檢查：
  4. burst 的實際段數與段長（相鄰缺口可能合併，要確認 k 有效）
  5. MNAR 是否真的偏好高振幅區段（否則它就退化成隨機）

產出：
  results/m3_rate_check.csv
  results/m3_burst_gap_stats.csv
  results/m3_mnar_selectivity.csv
  results/m3_rr_distribution.csv
  figures/m3_three_patterns.png      ← 放進 README 的那張圖
  figures/m3_burst_k_sweep.png
  results/m3_log.txt
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
from src import missingness as MS  # noqa: E402

RATES = [0.025, 0.05, 0.10, 0.20, 0.40]
K_SWEEP = [1, 2, 5, 10, 20]


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


def measure_rr(Xte: np.ndarray, fs: int) -> pd.DataFrame:
    """量測 RR 間期分布（k 的生理錨點，見 notes/process_and_decisions.md）。"""
    from scipy import signal as sps
    b, a = sps.butter(3, [5 / (fs / 2), 15 / (fs / 2)], btype="band")
    rr = []
    for i in range(len(Xte)):
        x = sps.filtfilt(b, a, Xte[i, :, 1].astype(np.float64))
        e = np.convolve(x ** 2, np.ones(12) / 12, mode="same")
        pk, _ = sps.find_peaks(e, distance=int(0.3 * fs), height=np.percentile(e, 90))
        if len(pk) >= 3:
            d = np.diff(pk) / fs
            d = d[(d > 0.3) & (d < 2.0)]
            if len(d) >= 2:
                rr.append(np.median(d))
    rr = np.array(rr)
    q = np.percentile(rr, [5, 25, 50, 75, 95])
    return pd.DataFrame([{
        "n_records_estimated": len(rr), "n_records_total": len(Xte),
        "rr_p5": q[0], "rr_p25": q[1], "rr_median": q[2], "rr_p75": q[3], "rr_p95": q[4],
        "hr_median_bpm": 60 / q[2],
    }])


def main() -> int:
    t0 = time.time()
    print("=" * 74)
    print(f"M3 — 缺失注入器驗證   {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 74)

    X, meta = D.load_full_cache()
    te = (meta.split == "test").values
    Xte = X[te]
    del X
    print(f"\n[0] 測試集：{Xte.shape}")
    T, C = Xte.shape[1], Xte.shape[2]

    # ---------------- RR 錨點 ----------------
    print("\n[0b] 量測 RR 間期分布（k 的生理錨點）")
    rr_df = measure_rr(Xte, config.SAMPLING_RATE)
    rr_df.to_csv(config.RESULTS_DIR / "m3_rr_distribution.csv", index=False)
    r = rr_df.iloc[0]
    print(f"    可估心律 {int(r.n_records_estimated)}/{int(r.n_records_total)} 筆；"
          f"RR 中位數 {r.rr_median:.2f}s（{r.hr_median_bpm:.0f} bpm），"
          f"p5–p95 = {r.rr_p5:.2f}–{r.rr_p95:.2f}s")

    sub = Xte[:300]   # 驗證用子集，300 筆足夠估計比例

    # ---------------- 檢查 1：缺失率 ----------------
    print("\n[1] 檢查一：實際缺失率（目標誤差 < 1%）")
    rows = []
    for pattern in MS.PATTERNS:
        for rate in RATES:
            kw = {"k": 5} if pattern == "burst" else {}
            M = MS.make_mask_batch(sub, pattern, rate, seed=0, **kw)
            actual = M.mean()
            rows.append({"pattern": pattern, "target_rate": rate,
                         "actual_rate": actual, "abs_error": abs(actual - rate)})
    rate_df = pd.DataFrame(rows)
    rate_df.to_csv(config.RESULTS_DIR / "m3_rate_check.csv", index=False)
    worst = rate_df.abs_error.max()
    for _, x in rate_df.iterrows():
        flag = "OK" if x.abs_error < 0.01 else "**超標**"
        print(f"    {x.pattern:8s} 目標 {x.target_rate:6.3f} → 實際 {x.actual_rate:.5f}"
              f"  誤差 {x.abs_error:.5f}  {flag}")
    print(f"    最大誤差 = {worst:.5f}  ({'全部通過' if worst < 0.01 else '**有型態未通過**'})")

    # ---------------- 檢查 2：可重現性 ----------------
    print("\n[2] 檢查二：可重現性（同 seed → 完全相同的遮罩）")
    ok_all = True
    for pattern in MS.PATTERNS:
        kw = {"k": 5} if pattern == "burst" else {}
        m1 = MS.make_mask_batch(sub[:50], pattern, 0.2, seed=42, **kw)
        m2 = MS.make_mask_batch(sub[:50], pattern, 0.2, seed=42, **kw)
        m3 = MS.make_mask_batch(sub[:50], pattern, 0.2, seed=43, **kw)
        same = np.array_equal(m1, m2)
        diff = not np.array_equal(m1, m3)
        ok_all &= (same and diff)
        print(f"    {pattern:8s} 同 seed 相同：{same}   不同 seed 相異：{diff}")
    print(f"    {'全部通過' if ok_all else '**未通過**'}")

    # ---------------- 檢查 4：burst 段數與段長 ----------------
    print("\n[3] 檢查：burst 的實際段數與段長（缺失率 20%）")
    rows = []
    for k in K_SWEEP:
        M = MS.make_mask_batch(sub[:200], "burst", 0.20, seed=0, k=k)
        ng, mg, mx = [], [], []
        for i in range(len(M)):
            for c in range(C):
                s = MS.gap_stats(M[i, :, c])
                ng.append(s["n_gaps"]); mg.append(s["mean_gap"]); mx.append(s["max_gap"])
        L_target = 0.20 * T / k
        rows.append({"k_target": k, "n_gaps_mean": np.mean(ng),
                     "gap_len_mean": np.mean(mg), "gap_len_max": np.max(mx),
                     "gap_len_target": L_target,
                     "gap_seconds": np.mean(mg) / config.SAMPLING_RATE,
                     "cardiac_cycles": (np.mean(mg) / config.SAMPLING_RATE) / r.rr_median})
    gap_df = pd.DataFrame(rows)
    gap_df.to_csv(config.RESULTS_DIR / "m3_burst_gap_stats.csv", index=False)
    print(gap_df.round(3).to_string(index=False))
    merged = (gap_df.n_gaps_mean < gap_df.k_target * 0.95).any()
    print(f"    相鄰缺口合併造成段數偏少：{'有，見上表' if merged else '無'}")

    # ---------------- 檢查 5：MNAR 選擇性 ----------------
    print("\n[4] 檢查：MNAR 是否真的偏好高振幅區段")
    rows = []
    for alpha in [0.0, 1.0, 2.0, 4.0]:
        M = MS.make_mask_batch(sub[:200], "mnar", 0.20, seed=0, alpha=alpha)
        amp_in, amp_out = [], []
        for i in range(len(M)):
            for c in range(C):
                x = np.abs(sub[i, :, c] - sub[i, :, c].mean())
                m = M[i, :, c]
                if m.any() and (~m).any():
                    amp_in.append(x[m].mean()); amp_out.append(x[~m].mean())
        ratio = np.mean(amp_in) / np.mean(amp_out)
        rows.append({"alpha": alpha, "amp_masked": np.mean(amp_in),
                     "amp_kept": np.mean(amp_out), "ratio": ratio})
        print(f"    alpha={alpha:.1f}: 被遮區段平均振幅 / 保留區段 = {ratio:.3f}")
    sel_df = pd.DataFrame(rows)
    sel_df.to_csv(config.RESULTS_DIR / "m3_mnar_selectivity.csv", index=False)
    base = sel_df.loc[sel_df.alpha == 0.0, "ratio"].iloc[0]
    used = sel_df.loc[sel_df.alpha == 2.0, "ratio"].iloc[0]
    print(f"    alpha=0（無選擇性）比值 {base:.3f}；alpha=2（採用值）比值 {used:.3f}"
          f"  → {'選擇性有效' if used > base * 1.1 else '**選擇性不足，需調整**'}")

    # ---------------- 檢查 3：三宮格圖 ----------------
    print("\n[5] 三宮格視覺化")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    idx, lead, rate = 0, 1, 0.20      # lead II
    x = Xte[idx, :, lead]
    t = np.arange(T) / config.SAMPLING_RATE
    specs = [("scatter", {}, "Scattered (MCAR)"),
             ("burst", {"k": 2}, "Burst, k=2 (MCAR, structured)"),
             ("mnar", {}, "Signal-dependent (MNAR)")]

    fig, axes = plt.subplots(3, 1, figsize=(11, 7.5), sharex=True, sharey=True)
    for ax, (pat, kw, title) in zip(axes, specs):
        rng = np.random.default_rng(0)
        m = MS.make_mask(T, C, pat, rate, rng,
                         signal=Xte[idx] if pat == "mnar" else None, **kw)[:, lead]
        ax.plot(t, x, lw=0.8, color="#1f4e79", zorder=2)
        d = np.diff(np.concatenate([[0], m.astype(np.int8), [0]]))
        for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
            ax.axvspan(t[s], t[min(e, T - 1)], color="#d62728", alpha=0.28, lw=0, zorder=1)
        st = MS.gap_stats(m)
        ax.set_title(f"{title}   —   {m.mean() * 100:.1f}% missing, "
                     f"{st['n_gaps']} gaps, longest {st['max_gap'] / config.SAMPLING_RATE:.2f}s",
                     fontsize=10, loc="left")
        ax.set_ylabel("mV")
        ax.grid(alpha=0.25, lw=0.4)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(f"Same record, same 20% missing rate — three missingness patterns "
                 f"(PTB-XL lead {config.LEAD_NAMES[lead]})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p = config.FIGURES_DIR / "m3_three_patterns.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print(f"    已存檔：{p}")

    # burst k 掃描圖
    fig, axes = plt.subplots(len(K_SWEEP), 1, figsize=(11, 9), sharex=True, sharey=True)
    for ax, k in zip(axes, K_SWEEP):
        rng = np.random.default_rng(0)
        m = MS.make_mask(T, C, "burst", rate, rng, k=k)[:, lead]
        ax.plot(t, x, lw=0.8, color="#1f4e79", zorder=2)
        d = np.diff(np.concatenate([[0], m.astype(np.int8), [0]]))
        for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
            ax.axvspan(t[s], t[min(e, T - 1)], color="#d62728", alpha=0.28, lw=0, zorder=1)
        L = 0.20 * T / k / config.SAMPLING_RATE
        ax.set_title(f"k={k}   gap ≈ {L:.2f}s ≈ {L / r.rr_median:.2f} cardiac cycles",
                     fontsize=9, loc="left")
        ax.grid(alpha=0.25, lw=0.4)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle("Fragmentation sweep — same 20% missing, split into k gaps", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p2 = config.FIGURES_DIR / "m3_burst_k_sweep.png"
    fig.savefig(p2, dpi=140)
    plt.close(fig)
    print(f"    已存檔：{p2}")

    print("\n" + "=" * 74)
    print(f"M3 驗證完成，總耗時 {time.time() - t0:.1f}s")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    log_path = config.RESULTS_DIR / "m3_log.txt"
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
