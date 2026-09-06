"""專案路徑與全域參數。

所有腳本都從這裡取路徑，不要在各處硬寫絕對路徑。
"""
from pathlib import Path

# 專案根目錄（本檔案所在位置）
PROJECT_ROOT = Path(__file__).resolve().parent

# PTB-XL 資料集位置。
# 預設假設資料集資料夾與本專案資料夾並排，例如：
#   side_project/
#     ├── ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/
#     └── ecg-missingness/          <- 本專案
# 若你把資料集搬到別處，只要改這一行。
PTBXL_ROOT = PROJECT_ROOT.parent / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"

# 產出目錄
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"
NOTES_DIR = PROJECT_ROOT / "notes"

for _d in (CACHE_DIR, RESULTS_DIR, FIGURES_DIR, NOTES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- 全域實驗參數 ---

# 開發階段的子集大小（plan_v3.md M1：開發先取 3000 筆，全量留到 M5 最後一次跑）
DEV_SUBSET_SIZE = 3000

# 隨機種子（M2 之後會用 SEEDS 跑多次，這裡是資料抽樣用的固定種子）
SEED = 42
SEEDS = [0, 1, 2]

# 訊號規格（records100 = 100Hz，10 秒，12 導程）
SAMPLING_RATE = 100
N_TIMESTEPS = 1000
N_CHANNELS = 12
LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF",
              "V1", "V2", "V3", "V4", "V5", "V6"]
