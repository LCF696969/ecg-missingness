"""M5c — 兩個收尾實驗。

(D) 碎裂曲線為什麼是非單調的？兩因子乘積模型
    H2（傷害 ∝ 被破壞的心搏比例）已被 M5b 否證：心搏破壞比例隨 k 單調上升
    （21% → 96%），但傷害在 k=10 達到最大後回落，Spearman 不顯著。
    H3：傷害由兩個互相拉扯的因子決定——
        (a) 受影響的範圍：隨 k 上升（缺口越多，碰到的心搏越多）
        (b) 每個缺口補得多爛：隨 k 下降（缺口越短，內插越準）
    兩者相乘應該在中間出現極大值。這裡直接檢驗這個乘積。

(E) SQI 守門機制在什麼條件下才有用？
    M5b 的異質實驗顯示守門幾乎沒有收益，但那個混合分布以低缺失率為主，
    而低缺失率本來就幾乎不傷害模型——沒有傷害，當然沒有東西可以守。
    這裡改用**高傷害混合**（較高缺失率、MNAR 佔多數）再測一次，
    回答「守門機制在什麼條件下值得部署」。

產出：
  results/m5c_mechanism.csv
  results/m5c_gate_highdamage.parquet
  results/m5c_gate_curves.csv
  results/m5c_log.txt
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
from scipy import stats as spstats  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

import config  # noqa: E402
from src import data as D  # noqa: E402
from src import features as F  # noqa: E402
from src import imputation as IMP  # noqa: E402
from src import missingness as MS  # noqa: E402

K_ALL = [1, 2, 5, 10, 20, 50, 100, 200]
RATE = 0.20
SEEDS = [0, 1, 2]

# (E) 高傷害混合：缺失率偏高、MNAR 佔多數
HD_RATES = [0.0, 0.05, 0.10, 0.20, 0.40]
HD_PROBS = [0.20, 0.15, 0.20, 0.25, 0.20]
HD_PATTERNS = ["scatter", "burst", "mnar"]
HD_PAT_PROBS = [0.2, 0.2, 0.6]


class Tee:
    def __init__(self, path: Path):
        self.file = open(path, "w", encoding="utf-8")

    def write(self, s):
        sys.__stdout__.write(s); self.file.write(s); self.file.flush()

    def flush(self):
        sys.__stdout__.flush(); self.file.flush()

    def close(self):
        self.file.close()


def main() -> int:
    t0 = time.time()
    print("=" * 78)
    print(f"M5c — 機制與守門   {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 78)

    X, meta = D.load_full_cache()
    is_te = (meta.split == "test").values
    Xte = X[is_te]; yte = meta.label.values[is_te]
    del X
    n, T, C = Xte.shape
    clf = joblib.load(PROJECT_ROOT / "models" / "m2_full_seed0.joblib")
    p_clean = clf.predict_proba(F.extract_features(Xte, fs=config.SAMPLING_RATE))[:, 1]
    auroc_clean = roc_auc_score(yte, p_clean)
    print(f"\n[0] 乾淨基線 AUROC = {auroc_clean:.4f}")

    # ================================================ (D) 兩因子乘積
    print("\n[D] 兩因子乘積模型（burst、20% 缺失、linear 補值）")
    sub = Xte[:400]
    bt = pd.read_csv(config.RESULTS_DIR / "m5b_beats_touched.csv").set_index("k")

    rmse_by_k = {}
    for k in K_ALL:
        M = MS.make_mask_batch(sub, "burst", RATE, seed=0, k=k)
        Xf = IMP.impute_batch(sub, M, "linear")
        rmse_by_k[k] = IMP.reconstruction_error(sub, Xf, M)["rmse"]

    runs = pd.read_parquet(config.RESULTS_DIR / "runs.parquet")
    d_old = (runs[(runs.group == "ksweep") & (runs.imputer == "linear")]
             .groupby("k").delta_auroc.mean())
    ext = pd.read_parquet(config.RESULTS_DIR / "m5b_ksweep_extended.parquet")
    d_new = ext[ext.imputer == "linear"].groupby("k").delta_auroc.mean()
    dmg = pd.concat([d_old, d_new]).sort_index()

    m = pd.DataFrame({
        "gap_len_pts": [RATE * T / k for k in K_ALL],
        "beats_touched": [bt.loc[k, "beats_touched_frac"] for k in K_ALL],
        "rmse_per_point": [rmse_by_k[k] for k in K_ALL],
        "damage": [-dmg.loc[k] for k in K_ALL],       # 轉成正的「傷害」
    }, index=pd.Index(K_ALL, name="k"))
    m["product"] = m.beats_touched * m.rmse_per_point

    print(m.round(4).to_string())
    for col in ["gap_len_pts", "beats_touched", "rmse_per_point", "product"]:
        r = spstats.spearmanr(m[col], m.damage)
        pear = spstats.pearsonr(m[col], m.damage)
        print(f"    {col:16s} vs 傷害：Spearman {r.statistic:+.3f} (p={r.pvalue:.3f})"
              f"   Pearson {pear.statistic:+.3f} (p={pear.pvalue:.3f})")
    m.to_csv(config.RESULTS_DIR / "m5c_mechanism.csv")

    # ================================================ (E) 高傷害下的守門
    print("\n[E] 高傷害混合下的守門機制")
    print(f"    缺失率 {dict(zip(HD_RATES, HD_PROBS))}；型態 {dict(zip(HD_PATTERNS, HD_PAT_PROBS))}")
    recs = []
    for seed in SEEDS:
        rng = np.random.default_rng(2000 + seed)
        rates_i = rng.choice(HD_RATES, size=n, p=HD_PROBS)
        pats_i = rng.choice(HD_PATTERNS, size=n, p=HD_PAT_PROBS)
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

        gmax = np.zeros(n); gn = np.zeros(n); flat = np.zeros(n); rough = np.zeros(n)
        hf = np.zeros(n)
        for i in range(n):
            g, c_ = 0, 0
            for c in range(C):
                s = MS.gap_stats(M[i, :, c]); g = max(g, s["max_gap"]); c_ += s["n_gaps"]
            gmax[i] = g; gn[i] = c_ / C
            d = np.diff(Xf[i], axis=0)
            flat[i] = (np.abs(d) < 1e-9).mean()
            rough[i] = np.abs(d).mean() / (np.median(np.abs(d)) + 1e-9)
            # 高頻能量比：補值造成的人工不連續會抬高高頻
            hf[i] = np.abs(np.diff(d, axis=0)).mean()
        recs.append(pd.DataFrame({
            "seed": seed, "true_rate": rates_i, "pattern": pats_i,
            "missing_frac": M.mean(axis=(1, 2)), "max_gap": gmax, "n_gaps": gn,
            "flat_frac": flat, "rough": rough, "hf_energy": hf,
            "prob": p, "label": yte, "abs_error": np.abs(p - yte),
        }))
    hd = pd.concat(recs, ignore_index=True)
    hd.to_parquet(config.RESULTS_DIR / "m5c_gate_highdamage.parquet", index=False)

    cands = ["missing_frac", "max_gap", "n_gaps", "flat_frac", "rough", "hf_energy"]
    print("\n    SQI 候選與預測誤差的 Spearman 相關：")
    for c in cands:
        r = spstats.spearmanr(hd[c], hd.abs_error)
        print(f"      {c:14s} rho = {r.statistic:+.4f}  (p={r.pvalue:.2e})")

    print("\n    覆蓋率-效能曲線（各候選 SQI 由好到壞排序）：")
    covs = [0.2, 0.4, 0.6, 0.8, 1.0]
    rows = []
    s0 = hd[hd.seed == 0].reset_index(drop=True)
    for c in cands + ["random"]:
        if c == "random":
            order = np.random.default_rng(0).permutation(len(s0))
        else:
            order = np.argsort(s0[c].values, kind="stable")
        line = []
        for cv in covs:
            idx = order[:int(len(s0) * cv)]
            yy, pp = s0.label.values[idx], s0.prob.values[idx]
            v = roc_auc_score(yy, pp) if len(np.unique(yy)) > 1 else np.nan
            line.append(v)
            rows.append({"sqi": c, "coverage": cv, "auroc": v})
        mono = all(line[i] >= line[i + 1] - 1e-4 for i in range(len(line) - 1))
        print(f"      {c:14s} " + "  ".join(f"{cv:.0%}:{v:.4f}" for cv, v in zip(covs, line))
              + f"   {'單調下降 → 可用' if mono else '非單調'}")
    pd.DataFrame(rows).to_csv(config.RESULTS_DIR / "m5c_gate_curves.csv", index=False)

    print("\n" + "=" * 78)
    print(f"M5c 完成，總耗時 {(time.time() - t0) / 60:.1f} 分鐘")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    log_path = config.RESULTS_DIR / "m5c_log.txt"
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
