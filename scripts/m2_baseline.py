"""M2 — 基線模型。

目的只有一個：拿到**無缺失時的天花板數字**，後面所有損失相對它計算。
同時記錄 3 個訓練 seed 的標準差，作為「效能下降是真的還是雜訊」的判斷基準。

產出：
  results/m2_baseline.csv        每個 seed 的 AUROC / AUPRC / 耗時
  results/m2_baseline_summary.csv  平均 ± 標準差
  results/m2_feature_importance.csv  特徵重要度（前 30）
  figures/m2_feature_importance.png
  data/cache/dev3000_feats.npy   特徵快取（M3–M5 重用）
  results/m2_log.txt
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
from src import features as F  # noqa: E402
from src import model as M  # noqa: E402


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


def main(tag: str = "dev3000") -> int:
    t0 = time.time()
    print("=" * 74)
    print(f"M2 — 基線模型   {datetime.now():%Y-%m-%d %H:%M:%S}   cache={tag}")
    print("=" * 74)

    # --- 1. 載入快取 ---
    X, meta = D.load_full_cache() if tag == "full" else D.load_cache(tag)
    print(f"\n[1] 載入快取：X={X.shape}, meta={meta.shape}")
    print(f"    split 分佈：{meta.split.value_counts().to_dict()}")

    # --- 2. 特徵 ---
    print(f"\n[2] 特徵抽取（每導程 {F.N_FEATURES_PER_LEAD} 個 × 12 導程）")
    t1 = time.time()
    feats = F.extract_features(X, fs=config.SAMPLING_RATE, verbose=True)
    del X
    print(f"    完成：{feats.shape}，耗時 {time.time() - t1:.1f}s")
    print(f"    NaN/Inf 檢查：{int(np.isnan(feats).sum())} / {int(np.isinf(feats).sum())}")
    np.save(config.CACHE_DIR / f"{tag}_feats.npy", feats)

    cols = F.feature_column_names(config.LEAD_NAMES)
    y = meta.label.values
    is_tr = (meta.split == "train").values
    is_va = (meta.split == "val").values
    is_te = (meta.split == "test").values
    print(f"    train/val/test = {is_tr.sum()}/{is_va.sum()}/{is_te.sum()}")

    # --- 3. 三個 seed ---
    print("\n[3] 訓練 3 個 seed（無缺失，天花板數字）")
    rows = []
    models = {}
    for seed in config.SEEDS:
        clf, elapsed, best_it = M.train_model(
            feats[is_tr], y[is_tr], feats[is_va], y[is_va], seed=seed)
        te = M.evaluate(clf, feats[is_te], y[is_te])
        va = M.evaluate(clf, feats[is_va], y[is_va])
        tr = M.evaluate(clf, feats[is_tr], y[is_tr])
        models[seed] = clf
        rows.append({
            "seed": seed,
            "test_auroc": te["auroc"], "test_auprc": te["auprc"],
            "val_auroc": va["auroc"], "val_auprc": va["auprc"],
            "train_auroc": tr["auroc"],
            "best_iteration": best_it,
            "fit_seconds": round(elapsed, 2),
        })
        print(f"    seed={seed}: test AUROC={te['auroc']:.4f}  AUPRC={te['auprc']:.4f}  "
              f"| val AUROC={va['auroc']:.4f} | train AUROC={tr['auroc']:.4f} "
              f"| best_iter={best_it} | {elapsed:.1f}s")

    df = pd.DataFrame(rows)
    df.to_csv(config.RESULTS_DIR / f"m2_{tag}_baseline.csv", index=False)

    summary = pd.DataFrame([{
        "metric": m,
        "mean": df[m].mean(),
        "std": df[m].std(ddof=1),
    } for m in ["test_auroc", "test_auprc", "val_auroc", "val_auprc", "train_auroc", "fit_seconds"]])
    summary.to_csv(config.RESULTS_DIR / f"m2_{tag}_baseline_summary.csv", index=False)

    print("\n[4] 彙總（3 seeds）")
    for _, r in summary.iterrows():
        print(f"    {r['metric']:14s}: {r['mean']:.4f} ± {r['std']:.4f}")

    # --- 5. 健檢：過擬合與洩漏 ---
    print("\n[5] 健檢")
    gap = df.train_auroc.mean() - df.test_auroc.mean()
    print(f"    train-test AUROC 落差：{gap:.4f}  "
          f"({'正常範圍' if gap < 0.25 else '偏大，注意過擬合'})")
    if df.test_auroc.mean() > 0.98:
        print("    ** 測試 AUROC > 0.98：這個數字不合理，必須回頭查資料洩漏 **")
    elif df.test_auroc.mean() > 0.93:
        print("    測試 AUROC > 0.93：偏高但仍可能合理（二元 NORM vs 異常比五類 macro 容易）")
    else:
        print("    測試 AUROC 落在文獻可比範圍內")

    # --- 6. 特徵重要度 ---
    print("\n[6] 特徵重要度（gain，三個 seed 平均）")
    imp = np.mean([m.feature_importances_ for m in models.values()], axis=0)
    imp_df = (pd.DataFrame({"feature": cols, "importance": imp})
              .sort_values("importance", ascending=False).reset_index(drop=True))
    imp_df.to_csv(config.RESULTS_DIR / f"m2_{tag}_feature_importance.csv", index=False)
    print(imp_df.head(15).to_string(index=False))

    # 依特徵「類型」彙總，看 Hjorth 是不是真的重要
    imp_df["ftype"] = [c.split("_", 1)[1] for c in imp_df.feature]
    by_type = (imp_df.groupby("ftype").importance.sum()
               .sort_values(ascending=False))
    print("\n    依特徵類型彙總（12 導程加總）：")
    for k, v in by_type.items():
        print(f"      {k:20s} {v:.4f}")
    by_type.to_csv(config.RESULTS_DIR / f"m2_{tag}_importance_by_type.csv")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 6))
    top = imp_df.head(25).iloc[::-1]
    ax.barh(top.feature, top.importance, color="#1f4e79")
    ax.set_xlabel("XGBoost gain importance (mean of 3 seeds)")
    ax.set_title("M2 baseline — top 25 features")
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / f"m2_{tag}_feature_importance.png", dpi=130)
    plt.close(fig)

    print("\n" + "=" * 74)
    print(f"M2 完成，總耗時 {time.time() - t0:.1f}s")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    TAG = sys.argv[1] if len(sys.argv) > 1 else "dev3000"
    log_path = config.RESULTS_DIR / f"m2_{TAG}_log.txt"
    tee = Tee(log_path)
    sys.stdout = tee
    try:
        code = main(tag=TAG)
    except Exception:
        print("\n**執行失敗**\n")
        traceback.print_exc(file=tee)
        code = 1
    finally:
        sys.stdout = sys.__stdout__
        tee.close()
    print(f"log 已寫入：{log_path}")
    sys.exit(code)
