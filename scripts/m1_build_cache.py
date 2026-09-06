"""M1 — 資料與標籤（在你的電腦上執行，因為 PTB-XL 原始資料在這裡）。

執行後會產生：
  results/m1_log.txt              執行紀錄（我會自動抓回去讀，你不用複製貼上）
  results/m1_label_distribution.csv   train/val/test 樣本數與正負比例表
  figures/m1_12lead_example.png   一張 12 導程心電圖
  data/cache/dev3000_X.npy        開發用 3000 筆波形快取 (N,1000,12) float32
  data/cache/dev3000_meta.csv     對應的標籤與切分

怎麼跑（任一種都可以）：
  1) 雙擊專案根目錄的 run_m1.bat
  2) 在 PyCharm 開啟 ecg-missingness 資料夾，右鍵這個檔案 -> Run
  3) 終端機：cd 到專案根目錄，執行  python scripts/m1_build_cache.py
"""
from __future__ import annotations

import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402


class Tee:
    """同時把輸出寫到螢幕與 log 檔，這樣 Claude 可以直接讀 log，不用你貼給他。"""

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
    print("=" * 70)
    print(f"M1 — 資料與標籤   開始時間 {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 70)

    # --- 環境檢查 ---
    print("\n[0] 環境檢查")
    print(f"    Python  : {sys.version.split()[0]}  ({sys.executable})")
    missing = []
    for mod in ["numpy", "pandas", "scipy", "matplotlib", "wfdb"]:
        try:
            m = __import__(mod)
            print(f"    {mod:12s}: {getattr(m, '__version__', '?')}")
        except ImportError:
            missing.append(mod)
            print(f"    {mod:12s}: **缺少**")
    if missing:
        print("\n缺少套件，請先執行：")
        print(f"    pip install {' '.join(missing)}")
        return 1

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from src import data as D  # noqa: E402

    print(f"    PTB-XL  : {config.PTBXL_ROOT}")
    if not config.PTBXL_ROOT.exists():
        print("    **找不到資料集資料夾**，請修改 config.py 的 PTBXL_ROOT")
        return 1

    # --- 1. metadata 與標籤 ---
    print("\n[1] 載入 metadata、映射 SCP 碼、建立二元標籤")
    df = D.load_metadata()
    df = D.add_split(df)
    print(f"    有診斷標籤的樣本數：{len(df)}")

    dist = D.label_distribution(df)
    dist_path = config.RESULTS_DIR / "m1_label_distribution.csv"
    dist.to_csv(dist_path, index=False)
    print("\n    標籤分佈表：")
    print(dist.to_string(index=False))
    print(f"\n    已存檔：{dist_path}")

    # 病患不跨集檢查（官方 strat_fold 應該已處理，這裡驗證一次）
    print("\n[1b] 驗證：同一病患是否跨越 train/val/test")
    pat_splits = df.groupby("patient_id")["split"].nunique()
    n_leak = int((pat_splits > 1).sum())
    print(f"    跨集的病患數：{n_leak}  ({'OK' if n_leak == 0 else '**注意，有洩漏風險**'})")

    # --- 2. 12 導程範例圖 ---
    print("\n[2] 畫一張 12 導程心電圖")
    import wfdb
    example_id = df.index[0]
    sig, meta_hdr = wfdb.rdsamp(str(config.PTBXL_ROOT / df.loc[example_id, "filename_lr"]))
    fig, axes = plt.subplots(12, 1, figsize=(11, 14), sharex=True)
    t = np.arange(sig.shape[0]) / config.SAMPLING_RATE
    for ch in range(12):
        axes[ch].plot(t, sig[:, ch], lw=0.7, color="#1f4e79")
        axes[ch].set_ylabel(config.LEAD_NAMES[ch], rotation=0, ha="right", va="center", fontsize=9)
        axes[ch].grid(alpha=0.25, lw=0.4)
        axes[ch].tick_params(labelsize=7)
    axes[-1].set_xlabel("time (s)")
    lbl = "abnormal" if df.loc[example_id, "label"] == 1 else "normal"
    fig.suptitle(f"PTB-XL ecg_id={example_id}  ({config.SAMPLING_RATE}Hz, 10s)  "
                 f"superclass={df.loc[example_id, 'superclass']}  label={lbl}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig_path = config.FIGURES_DIR / "m1_12lead_example.png"
    fig.savefig(fig_path, dpi=130)
    plt.close(fig)
    print(f"    已存檔：{fig_path}")

    # --- 3. 開發子集波形快取 ---
    print(f"\n[3] 抽開發子集（分層抽樣，n={config.DEV_SUBSET_SIZE}, seed={config.SEED}）並讀取波形")
    sub = D.make_dev_subset(df)
    print(f"    子集大小：{len(sub)}")
    sub_dist = D.label_distribution(sub)
    print(sub_dist.to_string(index=False))
    sub_dist.to_csv(config.RESULTS_DIR / "m1_dev_subset_distribution.csv", index=False)

    print("\n    開始讀取波形（約需 1-5 分鐘，視硬碟速度）...")
    t1 = time.time()
    X = D.load_waveforms(sub)
    print(f"    完成，shape={X.shape}, dtype={X.dtype}, 耗時 {time.time() - t1:.1f}s")
    print(f"    數值檢查：min={X.min():.3f}, max={X.max():.3f}, "
          f"mean={X.mean():.4f}, NaN={int(np.isnan(X).sum())}")

    npy_path, csv_path = D.save_cache(X, sub, tag="dev3000")
    size_mb = npy_path.stat().st_size / 1e6
    print(f"    已存檔：{npy_path}  ({size_mb:.1f} MB)")
    print(f"    已存檔：{csv_path}")

    print("\n" + "=" * 70)
    print(f"M1 完成，總耗時 {time.time() - t0:.1f}s")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    log_path = config.RESULTS_DIR / "m1_log.txt"
    tee = Tee(log_path)
    sys.stdout = tee
    try:
        code = main()
    except Exception:
        print("\n**執行失敗，錯誤訊息如下**\n")
        traceback.print_exc(file=tee)
        code = 1
    finally:
        sys.stdout = sys.__stdout__
        tee.close()
    print(f"\nlog 已寫入：{log_path}")
    if code != 0:
        print("執行失敗 — Claude 會讀 log 找原因，你不用複製貼上。")
    input("\n按 Enter 關閉視窗...")
    sys.exit(code)
