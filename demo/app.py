"""缺失結構的傷害地圖 — 互動 Demo

核心互動（plan_v3.md M6）：把缺失率拉過門檻時，介面會直接跳出
「此資料品質不足，預測不可信」——讓使用者自己體驗一次這個專案的論點。

在本機執行：
    streamlit run demo/app.py
部署：Streamlit Community Cloud，指向這個 repo 的 demo/app.py
（資料集不進 repo，demo 只用 40 筆預先存好的代表樣本）
"""
from __future__ import annotations

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
sys.path.insert(0, str(ROOT))

import joblib
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src import features as F
from src import imputation as IMP
from src import missingness as MS

FS = 100
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8a85", "#e3e3df"
PATTERN_C = {"scatter": "#2a78d6", "burst": "#eb6834", "mnar": "#1baf7a"}
PATTERN_ZH = {"scatter": "隨機散布 (MCAR)", "burst": "連續大段 (MCAR，有結構)",
              "mnar": "訊號相關 (MNAR)"}
PATTERN_EN = {"scatter": "scattered", "burst": "burst", "mnar": "signal-dependent"}
IMP_ZH = {"zero": "零填補 zero", "ffill": "前值填補 ffill", "linear": "線性內插 linear"}

st.set_page_config(page_title="缺失結構的傷害地圖", page_icon="📉", layout="wide")


@st.cache_resource
def load_model():
    """優先載入 XGBoost 原生 JSON 格式。

    joblib 的 pickle 綁定 xgboost / scikit-learn 的類別結構，換一個版本就可能載不起來，
    這是部署最常見的失敗點。XGBoost 的 JSON 格式是官方保證跨版本相容的序列化格式，
    所以 demo 以它為主、pickle 只當退路。兩條路徑在 40 筆樣本上的輸出機率完全相同
    （最大絕對差 0.0，驗證見 notes/process_and_decisions.md 的 P9）。
    """
    json_path = APP_DIR / "model.json"
    if json_path.exists():
        booster = xgb.Booster()
        booster.load_model(str(json_path))
        return booster
    return joblib.load(APP_DIR / "model.joblib")


def predict_abnormal(model, feat) -> float:
    """回傳「異常」的機率。同時支援 Booster 與 XGBClassifier。"""
    if isinstance(model, xgb.Booster):
        best = int(model.attributes().get("best_iteration", 0))
        return float(model.predict(xgb.DMatrix(feat),
                                   iteration_range=(0, best + 1))[0])
    return float(model.predict_proba(feat)[0, 1])


@st.cache_data
def load_samples():
    d = np.load(APP_DIR / "sample_pack.npz")
    X = d["X"].astype(np.float32) / 1000.0        # int16 µV -> mV
    meta = pd.DataFrame({k: d[k] for k in d.files if k != "X"})
    return X, meta


@st.cache_data
def load_harm():
    return pd.read_csv(APP_DIR / "harm_table.csv")


@st.cache_data
def clean_baseline():
    return 0.8894      # 完整 fold 10（2158 筆）上的無缺失 AUROC


def expected_harm(harm: pd.DataFrame, pattern: str, rate: float, imputer: str) -> float:
    """從實驗結果查「這個設定下預期的 AUROC 損失」（在量測過的缺失率之間線性內插）。"""
    s = harm[(harm.pattern == pattern) & (harm.imputer == imputer)].sort_values("rate")
    if s.empty:
        return 0.0
    return float(np.interp(rate, s.rate.values, -s.d.values))   # 回傳正的「損失」


def style(ax):
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8, length=0)


# ------------------------------------------------------------------ 側邊欄
X, meta = load_samples()
model = load_model()
harm = load_harm()
CLEAN = clean_baseline()

st.sidebar.header("設定")
labels = ["正常" if v == 0 else "異常" for v in meta.label]
opts = [f"#{i:02d}  ecg_id={e}  ({l})" for i, (e, l) in enumerate(zip(meta.ecg_id, labels))]
DEFAULT_SAMPLE = 10   # 乾淨時高信心判對，注入 MNAR 20% 會讓判讀翻轉——最能說明論點的一筆
sel = st.sidebar.selectbox("樣本（PTB-XL 測試集 fold 10）", range(len(opts)),
                           index=min(DEFAULT_SAMPLE, len(opts) - 1),
                           format_func=lambda i: opts[i])
lead = st.sidebar.selectbox("導程", range(12), index=1, format_func=lambda i: LEADS[i])

st.sidebar.divider()
pattern = st.sidebar.radio("缺失型態", list(PATTERN_ZH), format_func=lambda p: PATTERN_ZH[p])
rate = st.sidebar.slider("缺失率 (%)", 0.0, 40.0, 20.0, 2.5) / 100
k = 2
if pattern == "burst":
    k = st.sidebar.select_slider(
        "碎裂程度 k（缺口段數）", [1, 2, 5, 10, 20, 50],
        value=2,
        format_func=lambda v: f"k={v}  每段 {0.2 * 1000 / v / FS:.2f}s" if rate > 0 else f"k={v}")
imputer = st.sidebar.radio("補值方法", list(IMP_ZH), index=2, format_func=lambda m: IMP_ZH[m])
seed = st.sidebar.number_input("隨機種子", 0, 999, 0, 1)

st.sidebar.divider()
tol = st.sidebar.slider("可容忍的 AUROC 損失", 0.005, 0.10, 0.03, 0.005,
                        help="醫材規格上「還能用」的界線。預設 0.03 與門檻表一致。")

# ------------------------------------------------------------------ 計算
x_true = X[sel]
T, C = x_true.shape
rng = np.random.default_rng(int(seed))
if rate > 0:
    mask = MS.make_mask(T, C, pattern, rate, rng,
                        signal=x_true if pattern == "mnar" else None,
                        **({"k": k} if pattern == "burst" else {}))
else:
    mask = np.zeros((T, C), dtype=bool)
x_imp = IMP.impute(x_true, mask, imputer)

p_clean = predict_abnormal(model, F.extract_features(x_true[None]))
p_dirty = predict_abnormal(model, F.extract_features(x_imp[None]))
truth = int(meta.label.iloc[sel])

# 逐筆品質指標（不看標籤）
gap_stats = [MS.gap_stats(mask[:, c]) for c in range(C)]
max_gap = max(g["max_gap"] for g in gap_stats)
n_gaps = float(np.mean([g["n_gaps"] for g in gap_stats]))
loss = expected_harm(harm, pattern, rate, imputer)
usable = loss <= tol

# ------------------------------------------------------------------ 版面
st.title("缺失結構的傷害地圖")
st.caption("同樣缺 20% 的資料，缺法不同，模型受的傷差 37 倍。"
           "把左邊的缺失率往右拉，看模型什麼時候該閉嘴。")

if rate > 0:
    if usable:
        st.success(f"**資料品質足夠** — 在這個設定下，預期 AUROC 損失 **{loss:.3f}**，"
                   f"低於你設定的容忍上限 {tol:.3f}。預測可以採信。")
    else:
        st.error(f"**此資料品質不足，預測不可信** — 預期 AUROC 損失 **{loss:.3f}**，"
                 f"已超過容忍上限 {tol:.3f}。建議標記為不可用，而不是硬補後照常判讀。")

c1, c2 = st.columns([3, 2], gap="large")

with c1:
    st.subheader("訊號")
    t = np.arange(T) / FS
    fig, ax = plt.subplots(figsize=(8, 3.4))
    style(ax)
    ax.plot(t, x_true[:, lead], lw=1.8, color="#c9c9c4", zorder=1, label="original")
    xm = x_true[:, lead].astype(float).copy()
    xm[mask[:, lead]] = np.nan
    ax.plot(t, xm, lw=1.0, color="#1f4e79", zorder=3, label="observed")
    if rate > 0:
        ax.plot(t, x_imp[:, lead], lw=1.1, color="#e34948", ls="--", zorder=2, label="imputed")
        d = np.diff(np.concatenate([[0], mask[:, lead].astype(np.int8), [0]]))
        for s0, e0 in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
            ax.axvspan(t[s0], t[min(e0, T - 1)], color="#e34948", alpha=0.13, lw=0, zorder=0)
    ax.set_xlabel("time (s)", color=INK2, fontsize=9)
    ax.set_ylabel("mV", color=INK2, fontsize=9)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper right", labelcolor=INK)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    m1, m2, m3 = st.columns(3)
    m1.metric("實際缺失率", f"{mask.mean() * 100:.1f}%")
    m2.metric("最長缺口", f"{max_gap / FS:.2f} s",
              help="這一筆最長的連續缺口。它是守門機制中最有效的品質指標之一。")
    m3.metric("缺口段數", f"{n_gaps:.0f}")

with c2:
    st.subheader("模型判讀")
    verdict_dirty = "異常" if p_dirty > 0.5 else "正常"
    verdict_clean = "異常" if p_clean > 0.5 else "正常"
    truth_zh = "異常" if truth == 1 else "正常"

    a, b = st.columns(2)
    a.metric("有缺失時的判讀", verdict_dirty, f"信心 {max(p_dirty, 1 - p_dirty):.1%}")
    b.metric("同一筆、無缺失時", verdict_clean, f"信心 {max(p_clean, 1 - p_clean):.1%}")
    st.caption(f"真實標籤：**{truth_zh}**　·　異常機率 {p_clean:.3f} → {p_dirty:.3f}"
               f"（變動 {p_dirty - p_clean:+.3f}）")

    if verdict_clean != verdict_dirty:
        st.warning("**判讀在缺失後翻轉了。** 同一筆心電圖，只因為資料缺了一部分，"
                   "結論就從一邊跳到另一邊——而模型本身不會告訴你這件事發生了。")

    st.divider()
    st.subheader("這個設定落在傷害地圖的哪裡")
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    style(ax)
    ax.axhline(CLEAN, color=MUTED, lw=1.1, ls=(0, (4, 3)), zorder=1)
    for pat in ["scatter", "burst", "mnar"]:
        s = harm[(harm.pattern == pat) & (harm.imputer == imputer)].sort_values("rate")
        ax.plot(s.rate * 100, s.auroc, color=PATTERN_C[pat], lw=1.8,
                marker="o", ms=4, mec="white", mew=1.0,
                label=PATTERN_EN[pat], zorder=2,
                alpha=1.0 if pat == pattern else 0.35)
    if rate > 0:
        ax.scatter([rate * 100], [CLEAN - loss], s=130, marker="*",
                   color="#e34948", edgecolor="white", linewidth=1.2, zorder=5)
        ax.annotate("your setting", xy=(rate * 100, CLEAN - loss), xytext=(6, -12),
                    textcoords="offset points", color=INK, fontsize=8.5)
    ax.set_xlabel("missing rate (%)", color=INK2, fontsize=9)
    ax.set_ylabel("AUROC", color=INK2, fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="lower left", labelcolor=INK)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.caption(f"曲線來自 213 次評估（PTB-XL fold 10，2158 筆，固定模型）。"
               f"目前顯示補值方法：{IMP_ZH[imputer]}。")

st.divider()
with st.expander("這個 demo 在示範什麼？"):
    st.markdown("""
**一句話**：我在研究怎麼讓 AI 在資料品質不夠時說「我不知道」，而不是硬猜。

臨床儀器的缺失不是隨機的——機器暫停、警報、護理操作會造成**連續大段**的缺口，
病人躁動時電極接觸不良則會讓**訊號最強的地方最容易缺**。
但補值方法的評估文獻幾乎清一色用隨機遮蔽。

這個 demo 讓你自己試三種缺法。三個值得動手試試的設定：

1. **把型態切到「訊號相關 (MNAR)」，缺失率拉到 20%** —— 傷害是隨機散布的 37 倍。
   這是最貼近真實、也最少人測的一種。
2. **選「隨機散布」，補值方法切成「零填補」** —— 它是最差的選擇（損失 0.038）。
   但同樣是零填補，換到「連續大段」卻變成**最好**的選擇（損失 0.003）。
   **沒有一種補值方法在所有情況下都對。**
3. **選「連續大段」，把 k 從 1 拉到 50** —— 同樣缺 20%，傷害在中等碎裂程度時最大。
   這個非單調現象我們還沒有能解釋的機制（三個假設都被自己的資料否證了）。

完整結果、決策紀錄與被否證的假設：見 repo 的 `README.md` 與 `notes/process_and_decisions.md`。
""")
