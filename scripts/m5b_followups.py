"""M5b — 三個補強實驗，回應主實驗跑出來的問題。

(A) 延長碎裂掃描到 k = 50 / 100 / 200
    原本掃到 k=20，但那時缺口仍有 10 個取樣點，而散布型的缺口只有 1 點，
    兩者差 10 倍——所以「burst 應該收斂到 scatter」這個檢查其實沒做完。
    k=200 時（20% 缺失、1000 點）每段長度正好 = 1 點 = 散布型，收斂檢查才閉合。

(B) 機制檢驗：傷害是由「缺口多長」還是「碰到幾個心搏」決定？
    主實驗顯示傷害隨 k 非單調（k=10 最糟），與「越連續越糟」的原始假設不符。
    假設 H2：真正的驅動量是**被破壞的心搏比例**，而不是缺口長度。

(C) SQI 重新設計：主實驗的 SQI 相關性全部接近 0，原因是**設計缺陷**——
    我們對每一筆記錄注入完全相同的缺失率，所以逐筆之間根本沒有品質差異可排序。
    真實情況是逐筆變異很大：有些記錄幾乎乾淨、有些明顯有缺失。
    這裡改成每筆隨機抽一個缺失率與型態，SQI 才有東西可以分辨。

產出：
  results/m5b_ksweep_extended.parquet
  results/m5b_beats_touched.csv
  results/m5b_mixed_records.parquet
  results/m5b_sqi_correlation.csv
  results/m5b_log.txt
"""
from __future__ import annotations

import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import signal as sps  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

import config  # noqa: E402
from src import data as D  # noqa: E402
from src import features as F  # noqa: E402
from src import imputation as IMP  # noqa: E402
from src import missingness as MS  # noqa: E402

K_EXTENDED = [50, 100, 200]
IMPUTERS = ["zero", "ffill", "linear"]
SEEDS = [0, 1, 2]
RATE = 0.20

# (C) 逐筆缺失率的分布：多數記錄品質好、少數很差——逐筆變異大才是真實情況
MIX_RATES = [0.0, 0.025, 0.05, 0.10, 0.20, 0.40]
MIX_PROBS = [0.30, 0.25, 0.15, 0.15, 0.10, 0.05]


class Tee:
    def __init__(self, path: Path):
        self.file = open(path, "w", encoding="utf-8")

    def write(self, s):
        sys.__stdout__.write(s); self.file.write(s); self.file.flush()

    def flush(self):
        sys.__stdout__.flush(); self.file.flush()

    def close(self):
        self.file.close()


def detect_rpeaks(x: np.ndarray, fs: int) -> np.ndarray:
    b, a = sps.butter(3, [5 / (fs / 2), 15 / (fs / 2)], btype="band")
    e = np.convolve(sps.filtfilt(b, a, x.astype(np.float64)) ** 2,
                    np.ones(12) / 12, mode="same")
    pk, _ = sps.find_peaks(e, distance=int(0.3 * fs), height=np.percentile(e, 90))
    return pk


def main() -> int:
    t0 = time.time()
    print("=" * 78)
    print(f"M5b — 補強實驗   {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 78)

    X, meta = D.load_full_cache()
    is_te = (meta.split == "test").values
    Xte = X[is_te]; yte = meta.label.values[is_te]
    del X
    n, T, C = Xte.shape
    clf = joblib.load(PROJECT_ROOT / "models" / "m2_full_seed0.joblib")
    feats_clean = F.extract_features(Xte, fs=config.SAMPLING_RATE)
    p_clean = clf.predict_proba(feats_clean)[:, 1]
    auroc_clean = roc_auc_score(yte, p_clean)
    print(f"\n[0] 測試集 {Xte.shape}；乾淨基線 AUROC = {auroc_clean:.4f}")

    # ================================================== (A) 延長 k 掃描
    print(f"\n[A] 延長碎裂掃描 k = {K_EXTENDED}（缺失率 {RATE:.0%}）")
    rows = []
    for k in K_EXTENDED:
        for imputer in IMPUTERS:
            aur = []
            for seed in SEEDS:
                M = MS.make_mask_batch(Xte, "burst", RATE, seed=seed, k=k)
                Xf = IMP.impute_batch(Xte, M, imputer)
                p = clf.predict_proba(F.extract_features(Xf, fs=config.SAMPLING_RATE))[:, 1]
                a = roc_auc_score(yte, p)
                aur.append(a)
                rows.append({"group": "ksweep_ext", "pattern": "burst", "rate": RATE,
                             "k": k, "imputer": imputer, "inject_seed": seed,
                             "auroc": a, "auprc": average_precision_score(yte, p),
                             "delta_auroc": a - auroc_clean, "auroc_clean": auroc_clean})
            aur = np.array(aur)
            gl = RATE * T / k
            print(f"    k={k:<4d} (gap {gl:.0f} pts = {gl / config.SAMPLING_RATE:.3f}s) "
                  f"{imputer:7s} AUROC={aur.mean():.4f}±{aur.std(ddof=1):.4f} "
                  f"Δ={aur.mean() - auroc_clean:+.4f}")
    pd.DataFrame(rows).to_parquet(config.RESULTS_DIR / "m5b_ksweep_extended.parquet",
                                  index=False)

    # ================================================== (B) 心搏破壞比例
    print("\n[B] 機制檢驗：每個 k 破壞了多少比例的心搏（lead II，QRS 視窗 ±0.06s）")
    sub = Xte[:400]
    half = int(0.06 * config.SAMPLING_RATE)
    peaks = [detect_rpeaks(sub[i, :, 1], config.SAMPLING_RATE) for i in range(len(sub))]
    brows = []
    for k in [1, 2, 5, 10, 20, 50, 100, 200]:
        M = MS.make_mask_batch(sub, "burst", RATE, seed=0, k=k)
        frac, ngap_beat = [], []
        for i in range(len(sub)):
            pk = peaks[i]
            if len(pk) == 0:
                continue
            m = M[i, :, 1]
            hit = 0
            for r in pk:
                lo, hi = max(0, r - half), min(T, r + half + 1)
                if m[lo:hi].any():
                    hit += 1
            frac.append(hit / len(pk))
            ngap_beat.append(len(pk))
        brows.append({"k": k, "gap_len_pts": RATE * T / k,
                      "beats_touched_frac": float(np.mean(frac)),
                      "beats_per_record": float(np.mean(ngap_beat))})
        print(f"    k={k:<4d} gap={RATE * T / k:5.1f} pts → 被破壞的心搏比例 "
              f"{np.mean(frac) * 100:5.1f}%")
    bt = pd.DataFrame(brows)
    bt.to_csv(config.RESULTS_DIR / "m5b_beats_touched.csv", index=False)

    # 與傷害的關聯（用 linear 補值的 Δ）
    runs = pd.read_parquet(config.RESULTS_DIR / "runs.parquet")
    old = (runs[(runs.group == "ksweep") & (runs.imputer == "linear")]
           .groupby("k").delta_auroc.mean())
    new = (pd.DataFrame(rows).query("imputer == 'linear'").groupby("k").delta_auroc.mean())
    dmg = pd.concat([old, new]).sort_index()
    j = bt.set_index("k").join(dmg.rename("delta_auroc"))
    from scipy import stats as spstats
    r_len = spstats.spearmanr(j.gap_len_pts, j.delta_auroc)
    r_beat = spstats.spearmanr(j.beats_touched_frac, j.delta_auroc)
    print("\n    傷害(Δ AUROC, linear) 與兩個候選驅動量的 Spearman 相關：")
    print(f"      缺口長度      rho = {r_len.statistic:+.3f}  (p={r_len.pvalue:.3f})")
    print(f"      心搏破壞比例  rho = {r_beat.statistic:+.3f}  (p={r_beat.pvalue:.3f})")
    print(j.round(4).to_string())
    j.to_csv(config.RESULTS_DIR / "m5b_beats_vs_damage.csv")

    # ================================================== (C) 異質缺失下的 SQI
    print("\n[C] SQI 重新設計：逐筆隨機抽缺失率與型態")
    print(f"    缺失率分布 {dict(zip(MIX_RATES, MIX_PROBS))}")
    recs = []
    for seed in SEEDS:
        rng = np.random.default_rng(1000 + seed)
        rates_i = rng.choice(MIX_RATES, size=n, p=MIX_PROBS)
        pats_i = rng.choice(MS.PATTERNS, size=n)
        M = np.zeros((n, T, C), dtype=bool)
        for i in range(n):
            if rates_i[i] > 0:
                kw = {"k": 2} if pats_i[i] == "burst" else {}
                M[i] = MS.make_mask(T, C, pats_i[i], float(rates_i[i]), rng,
                                    signal=Xte[i] if pats_i[i] == "mnar" else None, **kw)
        Xf = IMP.impute_batch(Xte, M, "linear")
        p = clf.predict_proba(F.extract_features(Xf, fs=config.SAMPLING_RATE))[:, 1]
        a = roc_auc_score(yte, p)
        print(f"    seed={seed}: 全體 AUROC = {a:.4f} (Δ={a - auroc_clean:+.4f})，"
              f"平均缺失率 {M.mean():.3f}")

        gaps_max = np.zeros(n); gaps_n = np.zeros(n); flat = np.zeros(n); rough = np.zeros(n)
        for i in range(n):
            gm, gn = 0, 0
            for c in range(C):
                s = MS.gap_stats(M[i, :, c])
                gm = max(gm, s["max_gap"]); gn += s["n_gaps"]
            gaps_max[i] = gm; gaps_n[i] = gn / C
            d = np.diff(Xf[i], axis=0)
            flat[i] = (np.abs(d) < 1e-9).mean()
            rough[i] = np.abs(d).mean() / (np.median(np.abs(d)) + 1e-9)
        recs.append(pd.DataFrame({
            "seed": seed, "true_rate": rates_i, "pattern": pats_i,
            "missing_frac": M.mean(axis=(1, 2)), "max_gap": gaps_max,
            "n_gaps": gaps_n, "flat_frac": flat, "rough": rough,
            "prob": p, "label": yte, "abs_error": np.abs(p - yte),
        }))
    mix = pd.concat(recs, ignore_index=True)
    mix.to_parquet(config.RESULTS_DIR / "m5b_mixed_records.parquet", index=False)

    print("\n    SQI 候選與預測誤差的 Spearman 相關（異質缺失下）：")
    crows = []
    for c in ["missing_frac", "max_gap", "n_gaps", "flat_frac", "rough"]:
        rho = spstats.spearmanr(mix[c], mix.abs_error)
        crows.append({"sqi": c, "rho": rho.statistic, "p": rho.pvalue})
        print(f"      {c:14s} rho = {rho.statistic:+.4f}  (p={rho.pvalue:.2e})")
    pd.DataFrame(crows).to_csv(config.RESULTS_DIR / "m5b_sqi_correlation.csv", index=False)

    print("\n    覆蓋率-效能曲線（以 missing_frac 由小到大排序）：")
    s0 = mix[mix.seed == 0].reset_index(drop=True)
    order = np.lexsort((s0.max_gap.values, s0.missing_frac.values))
    for cv in [0.2, 0.4, 0.6, 0.8, 1.0]:
        idx = order[:int(len(s0) * cv)]
        print(f"      覆蓋 {cv:4.0%}: AUROC = {roc_auc_score(s0.label.values[idx], s0.prob.values[idx]):.4f}")

    print("\n" + "=" * 78)
    print(f"M5b 完成，總耗時 {(time.time() - t0) / 60:.1f} 分鐘")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    log_path = config.RESULTS_DIR / "m5b_log.txt"
    tee = Tee(log_path)
    sys.stdout = tee
    try:
        code = main()
    except Exception:
        print("\n**執行失敗**\n"); traceback.print_exc(file=tee); code = 1
    finally:
        sys.stdout = sys.__stdout__; tee.close()
    print(f"log 已寫入：{log_path}")
    sys.exit(code)
