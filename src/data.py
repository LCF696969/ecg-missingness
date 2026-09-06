"""M1：資料載入、SCP 碼映射、官方切分、波形快取。

設計原則：
- 標籤與切分邏輯集中在這裡，之後所有實驗共用，避免各腳本各自為政導致不一致。
- 切分用 PTB-XL 官方的 strat_fold（已處理同病患不跨集的問題）。
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402


# --------------------------------------------------------------------------
# metadata：載入 + SCP 碼映射到五大類 + 二元標籤
# --------------------------------------------------------------------------
def load_metadata(ptbxl_root: Path | None = None) -> pd.DataFrame:
    """讀 ptbxl_database.csv 與 scp_statements.csv，回傳含 superclass / label 的 DataFrame。

    label 定義（plan_v3.md）：二元分類，0 = 完全正常(NORM)，1 = 任何異常。
    只保留至少有一個 diagnostic SCP 碼的樣本。
    """
    root = Path(ptbxl_root) if ptbxl_root else config.PTBXL_ROOT

    db_path = root / "ptbxl_database.csv"
    scp_path = root / "scp_statements.csv"
    for p in (db_path, scp_path):
        if not p.exists():
            raise FileNotFoundError(
                f"找不到 {p}\n"
                f"請確認 config.PTBXL_ROOT 指向正確的 PTB-XL 資料夾（目前：{root}）"
            )

    df = pd.read_csv(db_path, index_col="ecg_id")
    df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)  # 字串轉 dict

    agg = pd.read_csv(scp_path, index_col=0)
    agg = agg[agg.diagnostic == 1]  # 只留診斷類的碼

    def to_superclass(scp_dict: dict) -> list[str]:
        return sorted({agg.loc[c, "diagnostic_class"] for c in scp_dict if c in agg.index})

    df["superclass"] = df.scp_codes.apply(to_superclass)
    df = df[df.superclass.map(len) > 0].copy()  # 丟掉無診斷標籤的
    df["label"] = (~df.superclass.apply(lambda s: s == ["NORM"])).astype(int)

    # 方便之後看五大類分佈
    for cls in ["NORM", "CD", "MI", "HYP", "STTC"]:
        df[f"has_{cls}"] = df.superclass.apply(lambda s, c=cls: c in s).astype(int)

    return df


def add_split(df: pd.DataFrame) -> pd.DataFrame:
    """依官方 strat_fold 加上 split 欄位：train(1-8) / val(9) / test(10)。"""
    df = df.copy()
    df["split"] = np.select(
        [df.strat_fold <= 8, df.strat_fold == 9, df.strat_fold == 10],
        ["train", "val", "test"],
        default="unknown",
    )
    return df


def label_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """train/val/test 的樣本數與正負類比例表（M1 完成判定之一）。"""
    rows = []
    for split in ["train", "val", "test", "ALL"]:
        sub = df if split == "ALL" else df[df.split == split]
        n = len(sub)
        n_pos = int(sub.label.sum())
        rows.append({
            "split": split,
            "n": n,
            "n_normal(label=0)": n - n_pos,
            "n_abnormal(label=1)": n_pos,
            "abnormal_ratio": round(n_pos / n, 4) if n else np.nan,
            **{f"has_{c}": int(sub[f"has_{c}"].sum()) for c in ["NORM", "CD", "MI", "HYP", "STTC"]},
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 開發子集抽樣
# --------------------------------------------------------------------------
def make_dev_subset(df: pd.DataFrame, n: int | None = None, seed: int | None = None) -> pd.DataFrame:
    """抽開發用子集，依 (split, label) 分層，維持原始比例。

    分層抽樣而不是取前 n 筆，避免 ecg_id 排序帶進任何系統性偏誤。
    """
    n = n or config.DEV_SUBSET_SIZE
    seed = seed if seed is not None else config.SEED
    if n >= len(df):
        return df.copy()

    frac = n / len(df)
    rng = np.random.RandomState(seed)
    parts = []
    for _, grp in df.groupby(["split", "label"], sort=True):
        k = max(1, int(round(len(grp) * frac)))
        idx = rng.choice(len(grp), size=min(k, len(grp)), replace=False)
        parts.append(grp.iloc[np.sort(idx)])
    out = pd.concat(parts).sort_index()
    return out


# --------------------------------------------------------------------------
# 波形讀取與快取
# --------------------------------------------------------------------------
def load_waveforms(sub_df: pd.DataFrame, ptbxl_root: Path | None = None,
                   verbose: bool = True) -> np.ndarray:
    """讀 records100 波形，回傳 (N, 1000, 12) 的 float32 陣列。"""
    import wfdb  # 延後 import，讓沒裝 wfdb 時錯誤訊息更清楚

    root = Path(ptbxl_root) if ptbxl_root else config.PTBXL_ROOT
    sigs = []
    total = len(sub_df)
    for i, fn in enumerate(sub_df.filename_lr):
        sig, _ = wfdb.rdsamp(str(root / fn))
        sigs.append(sig)
        if verbose and (i + 1) % 500 == 0:
            print(f"    讀取波形 {i + 1}/{total} ...", flush=True)
    arr = np.stack(sigs).astype(np.float32)
    return arr


def save_cache(X: np.ndarray, meta: pd.DataFrame, tag: str) -> tuple[Path, Path]:
    """存成 npy + csv，回傳 (npy 路徑, csv 路徑)。"""
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    npy_path = config.CACHE_DIR / f"{tag}_X.npy"
    csv_path = config.CACHE_DIR / f"{tag}_meta.csv"
    np.save(npy_path, X)
    keep = ["patient_id", "age", "sex", "strat_fold", "split", "label",
            "has_NORM", "has_CD", "has_MI", "has_HYP", "has_STTC", "filename_lr"]
    meta[[c for c in keep if c in meta.columns]].to_csv(csv_path)
    return npy_path, csv_path


def load_cache(tag: str) -> tuple[np.ndarray, pd.DataFrame]:
    """載入快取。"""
    X = np.load(config.CACHE_DIR / f"{tag}_X.npy")
    meta = pd.read_csv(config.CACHE_DIR / f"{tag}_meta.csv", index_col="ecg_id")
    assert len(X) == len(meta), f"快取不一致：X={len(X)}, meta={len(meta)}"
    return X, meta


def load_full_cache(as_float32: bool = True) -> tuple[np.ndarray, pd.DataFrame]:
    """載入 M1b 建立的全量分塊快取，回傳 (N,1000,12)。

    磁碟上是 int16（單位 µV，無損）；預設轉回 float32 的 mV，與 dev 快取一致。
    """
    import json

    info = json.loads((config.CACHE_DIR / "full_info.json").read_text(encoding="utf-8"))
    parts = [np.load(config.CACHE_DIR / c["file"]) for c in info["chunks"]]
    X = np.concatenate(parts, axis=0)
    if as_float32:
        X = (X.astype(np.float32) * np.float32(info["scale_to_mV"]))
    meta = pd.read_csv(config.CACHE_DIR / "full_meta.csv", index_col="ecg_id")
    assert len(X) == len(meta) == info["n_records"], (
        f"快取不一致：X={len(X)}, meta={len(meta)}, info={info['n_records']}")
    return X, meta
