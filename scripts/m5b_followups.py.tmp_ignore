"""M5 — 主實驗（情境 A：訓練用乾淨資料，測試注入缺失）。

設計要點（見 notes/process_and_decisions.md 的 Q1 與 F1）：
- **固定模型**：使用 M2 訓練好的 seed=0 模型，不重新訓練。
  因此誤差帶反映的是「缺口剛好落在哪裡」的隨機性，而不是模型訓練的隨機性。
- **配對比較**：乾淨與受損是同一批測試樣本、同一個模型，因此 ΔAUROC 用配對 bootstrap
  估信賴區間，比兩個獨立估計緊得多。

實驗規模：
  主網格   5 缺失率 × 3 型態 × 3 補值 × 3 注入 seed = 135
  碎裂掃描 5 個 k × 3 補值 × 3 注入 seed          =  45
  合計 180 次評估（不含乾淨基線）

產出：
  results/runs.parquet          每次評估一行（逐次追加，中途當掉不會全丟）
  results/m5_sqi_records.parquet 逐筆的 SQI 特徵與預測正確與否
  results/m5_log.txt
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
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

import config  # noqa: E402
from src import data as D  # noqa: E402
from src import features as F  # noqa: E402
from src import imputation as IMP  # noqa: E402
from src import missingness as MS  # noqa: E402

RATES = [0.025, 0.05, 0.10, 0.20, 0.40]
PATTERNS = ["scatter", "burst", "mnar"]
IMPUTERS = ["zero", "ffill", "linear"]
INJECT_SEEDS = [0, 1, 2]
K_SWEEP = [1, 2, 5, 10, 20]
K_DEFAULT = 2          # 主網格中 burst 的預設碎裂程度
K_SWEEP_RATE = 0.20    # 碎裂掃描固定的缺失率


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


# ------------------------------------------------------------------ SQI
def sqi_features(M: np.ndarray, Xf: np.ndarray) -> pd.DataFrame:
    """逐筆計算**不看標籤**的訊號品質指標。

    候選特徵（全部只用遮罩與補值後的訊號，不用真值、不用標籤）：
      missing_frac  缺失總比例
      max_gap       最長連續缺口（取樣點），跨導程取最大
      n_gaps        缺口段數（跨導程平均）
      flat_frac     補值後訊號「完全平坦」的比例（前值填補與零填補的特徵痕跡）
      rough         補值後一階差分的異常程度（相對於該筆自身的尺度）
    """
    n, T, C = M.shape
    rows = []
    for i in range(n):
        gaps_max, gaps_n = 0, 0
        for c in range(C):
            s = MS.gap_stats(M[i, :, c])
            gaps_max = max(gaps_max, s["max_gap"])
            gaps_n += s["n_gaps"]
        d = np.diff(Xf[i], axis=0)
        scale = np.median(np.abs(d)) + 1e-9
        rows.append({
            "missing_frac": float(M[i].mean()),
            "max_gap": int(gaps_max),
            "n_gaps": gaps_n / C,
            "flat_frac": float((np.abs(d) < 1e-9).mean()),
            "rough": float(np.abs(d).mean() / scale),
        })
    return pd.DataFrame(rows)


def paired_bootstrap_delta(y, p_clean, p_dirty, n_boot=2000, seed=0):
    """配對 bootstrap：同一批樣本索引同時用於乾淨與受損，估 ΔAUROC 的信賴區間。"""
    rng = np.random.default_rng(seed)
    n = len(y)
    d = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        d.append(roc_auc_score(y[idx], p_dirty[idx]) - roc_auc_score(y[idx], p_clean[idx]))
    d = np.array(d)
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main() -> int:
    t0 = time.time()
    print("=" * 78)
    print(f"M5 — 主實驗   {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 78)

    X, meta = D.load_full_cache()
    is_te = (meta.split == "test").values
    Xte = X[is_te]
    yte = meta.label.values[is_te]
    del X
    n, T, C = Xte.shape
    print(f"\n[0] 測試集 {Xte.shape}，正類 {int(yte.sum())} / 負類 {int((~yte.astype(bool)).sum())}")

    clf = joblib.load(PROJECT_ROOT / "models" / "m2_full_seed0.joblib")
    print("    已載入固定模型 models/m2_full_seed0.joblib（train seed=0，不重新訓練）")

    # ---------- 乾淨基線 ----------
    feats_clean = F.extract_features(Xte, fs=config.SAMPLING_RATE)
    p_clean = clf.predict_proba(feats_clean)[:, 1]
    auroc_clean = roc_auc_score(yte, p_clean)
    auprc_clean = average_precision_score(yte, p_clean)
    print(f"\n[1] 乾淨基線：AUROC = {auroc_clean:.4f}   AUPRC = {auprc_clean:.4f}")

    runs_path = config.RESULTS_DIR / "runs.parquet"
    rows: list[dict] = []
    sqi_rows: list[pd.DataFrame] = []

    def run_cell(pattern, rate, imputer, seed, k, tag):
        t1 = time.time()
        kw = {"k": k} if pattern == "burst" else {}
        M = MS.make_mask_batch(Xte, pattern, rate, seed=seed, **kw)
        Xf = IMP.impute_batch(Xte, M, imputer)
        feats = F.extract_features(Xf, fs=config.SAMPLING_RATE)
        p = clf.predict_proba(feats)[:, 1]
        auroc = roc_auc_score(yte, p)
        auprc = average_precision_score(yte, p)
        dmean, dlo, dhi = paired_bootstrap_delta(yte, p_clean, p, n_boot=1000, seed=seed)
        rec = {
            "group": tag, "pattern": pattern, "rate": rate, "k": k,
            "imputer": imputer, "inject_seed": seed,
            "auroc": auroc, "auprc": auprc,
            "delta_auroc": auroc - auroc_clean,
            "delta_auroc_boot_mean": dmean,
            "delta_ci_lo": dlo, "delta_ci_hi": dhi,
            "actual_missing_rate": float(M.mean()),
            "runtime_s": round(time.time() - t1, 2),
        }
        rows.append(rec)
        pd.DataFrame(rows).to_parquet(runs_path, index=False)   # 逐次落地

        if tag == "main" and seed == 0:
            s = sqi_features(M, Xf)
            s["pattern"] = pattern; s["rate"] = rate; s["imputer"] = imputer
            s["prob"] = p; s["label"] = yte
            s["abs_error"] = np.abs(p - yte)
            s["correct"] = ((p > 0.5).astype(int) == yte).astype(int)
            sqi_rows.append(s)
        return rec

    # ---------- 主網格 ----------
    total = len(RATES) * len(PATTERNS) * len(IMPUTERS) * len(INJECT_SEEDS)
    print(f"\n[2] 主網格 {total} 次評估（burst 固定 k={K_DEFAULT}）")
    done = 0
    for pattern in PATTERNS:
        for rate in RATES:
            for imputer in IMPUTERS:
                for seed in INJECT_SEEDS:
                    r = run_cell(pattern, rate, imputer, seed, K_DEFAULT, "main")
                    done += 1
                    if seed == INJECT_SEEDS[-1]:
                        sel = [x for x in rows if x["group"] == "main"
                               and x["pattern"] == pattern and x["rate"] == rate
                               and x["imputer"] == imputer]
                        a = np.array([x["auroc"] for x in sel])
                        print(f"    [{done:3d}/{total}] {pattern:8s} rate={rate:5.3f} "
                              f"{imputer:7s}  AUROC={a.mean():.4f}±{a.std(ddof=1):.4f}  "
                              f"Δ={a.mean() - auroc_clean:+.4f}  ({r['runtime_s']:.1f}s/次)")

    # ---------- 碎裂掃描 ----------
    total_k = len(K_SWEEP) * len(IMPUTERS) * len(INJECT_SEEDS)
    print(f"\n[3] 碎裂掃描 {total_k} 次評估（burst、缺失率 {K_SWEEP_RATE:.0%}）")
    done = 0
    for k in K_SWEEP:
        for imputer in IMPUTERS:
            for seed in INJECT_SEEDS:
                run_cell("burst", K_SWEEP_RATE, imputer, seed, k, "ksweep")
                done += 1
            sel = [x for x in rows if x["group"] == "ksweep"
                   and x["k"] == k and x["imputer"] == imputer]
            a = np.array([x["auroc"] for x in sel])
            print(f"    [{done:3d}/{total_k}] k={k:<3d} {imputer:7s}  "
                  f"AUROC={a.mean():.4f}±{a.std(ddof=1):.4f}  Δ={a.mean() - auroc_clean:+.4f}")

    # ---------- 落地 ----------
    df = pd.DataFrame(rows)
    df["auroc_clean"] = auroc_clean
    df["auprc_clean"] = auprc_clean
    df.to_parquet(runs_path, index=False)
    print(f"\n[4] 已存檔：{runs_path}（{len(df)} 列）")

    sqi = pd.concat(sqi_rows, ignore_index=True)
    sqi.to_parquet(config.RESULTS_DIR / "m5_sqi_records.parquet", index=False)
    print(f"    已存檔：m5_sqi_records.parquet（{len(sqi)} 列）")

    # ---------- 特徵位移分析（檢驗 H1：Hjorth 是否對破壞最敏感）----------
    print("\n[5] 特徵位移分析（檢驗 H1）")
    cols = F.feature_column_names(config.LEAD_NAMES)
    imp = pd.Series(clf.feature_importances_, index=cols)
    shift_rows = []
    for pattern in PATTERNS:
        M = MS.make_mask_batch(Xte, pattern, 0.20, seed=0,
                               **({"k": K_DEFAULT} if pattern == "burst" else {}))
        Xf = IMP.impute_batch(Xte, M, "linear")
        fd = F.extract_features(Xf, fs=config.SAMPLING_RATE)
        sd = np.abs(fd.mean(axis=0) - feats_clean.mean(axis=0)) / (feats_clean.std(axis=0) + 1e-9)
        s = pd.Series(sd, index=cols)
        for ftype in sorted({c.split("_", 1)[1] for c in cols}):
            sel = [c for c in cols if c.split("_", 1)[1] == ftype]
            shift_rows.append({"pattern": pattern, "ftype": ftype,
                               "mean_shift": float(s[sel].mean()),
                               "importance": float(imp[sel].sum()),
                               "damage_contrib": float((s[sel] * imp[sel]).sum())})
    sh = pd.DataFrame(shift_rows)
    sh.to_csv(config.RESULTS_DIR / "m5_feature_shift.csv", index=False)
    piv = sh.pivot(index="ftype", columns="pattern", values="mean_shift")
    piv["importance"] = sh.groupby("ftype").importance.first()
    print(piv.sort_values("burst", ascending=False).round(3).to_string())

    print("\n" + "=" * 78)
    print(f"M5 完成，總耗時 {(time.time() - t0) / 60:.1f} 分鐘")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    log_path = config.RESULTS_DIR / "m5_log.txt"
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
