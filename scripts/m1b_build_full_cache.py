"""M1b — 建立全量波形快取（在你的電腦上執行一次，約 8-10 分鐘）。

為什麼要做：M2 的 bootstrap 顯示，開發子集的測試集只有 303 筆時，
AUROC 的抽樣不確定性 SE≈0.020，是 seed 標準差(0.0047)的 4 倍。
在那個雜訊水準下，M5 主圖的三條線會分不開。用完整的 fold 10 (2158 筆)
可把 SE 降到約 0.0075，主圖才有意義。

儲存格式：int16，單位 µV（原始資料解析度就是 1µV/LSB，實測 round-trip 誤差為 0，
完全無損），大小是 float32 的一半。分塊儲存以便搬運。

產出：
  data/cache/full_meta.csv
  data/cache/full_info.json          scale、分塊大小、dtype
  data/cache/full_X_chunk{0,1,2}.npy int16, µV
  results/m1b_log.txt
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402

N_CHUNKS = 3
SCALE = 1000  # mV -> µV


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
    print(f"M1b — 全量波形快取   {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 74)

    import numpy as np
    import wfdb
    from src import data as D

    print("\n[1] 載入 metadata 與標籤")
    df = D.load_metadata()
    df = D.add_split(df)
    print(f"    樣本數：{len(df)}")
    print(f"    split：{df.split.value_counts().to_dict()}")

    keep = ["patient_id", "age", "sex", "strat_fold", "split", "label",
            "has_NORM", "has_CD", "has_MI", "has_HYP", "has_STTC", "filename_lr"]
    meta_path = config.CACHE_DIR / "full_meta.csv"
    df[[c for c in keep if c in df.columns]].to_csv(meta_path)
    print(f"    已存檔：{meta_path}")

    print(f"\n[2] 讀取全部 {len(df)} 筆波形並分 {N_CHUNKS} 塊存成 int16(µV)")
    idx_splits = np.array_split(np.arange(len(df)), N_CHUNKS)
    files = df.filename_lr.values
    total_clipped = 0
    max_abs_mv = 0.0
    chunk_meta = []

    for ci, idx in enumerate(idx_splits):
        t1 = time.time()
        buf = np.empty((len(idx), config.N_TIMESTEPS, config.N_CHANNELS), dtype=np.int16)
        for j, i in enumerate(idx):
            sig, _ = wfdb.rdsamp(str(config.PTBXL_ROOT / files[i]))
            m = float(np.abs(sig).max())
            if m > max_abs_mv:
                max_abs_mv = m
            q = np.round(sig * SCALE)
            n_clip = int((np.abs(q) > 32767).sum())
            if n_clip:
                total_clipped += n_clip
                q = np.clip(q, -32767, 32767)
            buf[j] = q.astype(np.int16)
            if (j + 1) % 2000 == 0:
                print(f"    chunk {ci}: {j + 1}/{len(idx)} ...", flush=True)

        out = config.CACHE_DIR / f"full_X_chunk{ci}.npy"
        np.save(out, buf)
        size_mb = out.stat().st_size / 1e6
        chunk_meta.append({"chunk": ci, "n": int(len(idx)),
                           "start": int(idx[0]), "end": int(idx[-1]) + 1,
                           "file": out.name, "size_mb": round(size_mb, 1)})
        print(f"    chunk {ci} 完成：{buf.shape} -> {out.name} ({size_mb:.1f} MB), "
              f"耗時 {time.time() - t1:.1f}s")
        del buf

    print(f"\n    全資料最大絕對振幅：{max_abs_mv:.3f} mV")
    print(f"    因超出 int16 範圍而被截斷的取樣點數：{total_clipped} "
          f"({'無失真' if total_clipped == 0 else '**注意**'})")

    info = {
        "n_records": int(len(df)),
        "n_timesteps": config.N_TIMESTEPS,
        "n_channels": config.N_CHANNELS,
        "dtype": "int16",
        "unit": "microvolt",
        "scale_to_mV": 1.0 / SCALE,
        "chunks": chunk_meta,
        "max_abs_mV": round(max_abs_mv, 4),
        "clipped_samples": int(total_clipped),
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    info_path = config.CACHE_DIR / "full_info.json"
    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"    已存檔：{info_path}")

    print("\n[3] 驗證：重新載入第一塊，檢查與原始檔一致")
    chk = np.load(config.CACHE_DIR / "full_X_chunk0.npy")
    sig0, _ = wfdb.rdsamp(str(config.PTBXL_ROOT / files[0]))
    back = chk[0].astype(np.float64) / SCALE
    err = float(np.abs(back - sig0).max())
    print(f"    第 1 筆 round-trip 最大誤差：{err:.8f} mV  "
          f"({'完全無損' if err == 0 else '有誤差，需檢查'})")

    print("\n" + "=" * 74)
    print(f"M1b 完成，總耗時 {time.time() - t0:.1f}s")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    log_path = config.RESULTS_DIR / "m1b_log.txt"
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
    print(f"\nlog 已寫入：{log_path}")
    input("\n按 Enter 關閉...")
    sys.exit(code)
