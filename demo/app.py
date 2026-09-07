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

tab1, tab2, tab3, tab4 = st.tabs(["怎麼操作", "怎麼看數字", "我們量到什麼", "別誤讀"])

# ─────────────────────────────────────────────── 怎麼操作
with tab1:
    st.markdown("""
左邊每一個控制項，先講**白話**，再講**技術**。

| 控制項 | 白話 | 技術 |
|---|---|---|
| **樣本** | 換一張心電圖。括號裡是這張圖的**正確答案**，所以你隨時知道模型有沒有答對 | 40 筆代表樣本，全部來自 fold 10 測試集，模型訓練時沒看過 |
| **導程** | 換一個角度看同一顆心臟。**只影響你看到的線，不影響模型的判斷** | 純視覺化選擇；推論始終用完整 12 導程 |
| **缺失型態** | 資料是「怎麼」缺的。這是最重要的一個旋鈕 | 見下方說明 |
| **缺失率** | 總共缺掉多少 | 固定遮掉 `round(p·T)` 個點，不是每點獨立丟硬幣——後者會讓實際缺失率帶抽樣波動 |
| **碎裂程度 k** | 同樣缺這麼多，是集中一大塊還是散成好幾小塊 | 只有「連續大段」時出現。段長平均分配，總缺失量精確 |
| **補值方法** | 怎麼把缺掉的地方填回去 | 三種都只碰缺失點，**觀測點改動數 = 0** |
| **隨機種子** | 缺失的**位置**重抽一次。缺多少不變，只是缺在別的地方 | 換幾個看看結果穩不穩——這正是主圖誤差帶反映的隨機源 |
| **可容忍的 AUROC 損失** | **你**願意接受多少誤差。它決定上方是綠燈還是紅燈 | 見「怎麼看數字」分頁 |

#### 三種缺失型態實際在做什麼

- **隨機散布**：每個時間點獨立地掉。像訊號傳輸偶爾掉封包。
- **連續大段**：缺失集中成 k 段連續的空白。像機器暫停、警報、護理操作打斷記錄。
- **訊號相關**：缺失機率**隨該段訊號振幅上升**——訊號越強的地方越容易缺。
  在心電圖上，振幅最大的地方就是 QRS 波，也就是診斷資訊最集中的地方。

> ⚠️ 前兩種在統計上**是同一種機制**（Rubin 分類的 MCAR），差別只在空間結構；
> 只有第三種才是 MNAR。這個區分很重要，下面「別誤讀」有說明。
>
> **第三種是刻意設計的壓力測試情境**，用來問「如果缺失剛好挑最重要的地方發生會怎樣」。
> 我沒有在真實臨床資料上量過它實際發生的頻率。

#### 三個值得親手拉一次的設定

1. **樣本 #10 · 訊號相關 MNAR · 20%** — 判讀會當場翻轉。
2. **隨機散布 + 零填補**（損失 0.038）**→ 切到連續大段**（損失 0.003）— 同一個補值方法，最差變最好。
3. **連續大段 · 20% · k 從 1 拉到 50** — 傷害不是單調的，中間最糟。
""")

# ─────────────────────────────────────────────── 怎麼看數字
with tab2:
    st.markdown("""
#### 先講 AUROC 是什麼

模型對每一筆會輸出一個「異常機率」。**AUROC 問的是：隨機抽一個真的異常、一個真的正常，
模型給異常那筆的分數比較高的機率有多大。**

- **1.0** = 完美排序　　**0.5** = 跟丟硬幣一樣　　**本專案乾淨基線 = 0.8894**
- 它**只看排序**，不受你把判定門檻設在 0.5 還是 0.3 影響——所以拿來衡量「資料變差讓模型退步多少」很乾淨。

#### 畫面上每個數字怎麼看

| 數字 | 方向 | 怎麼判讀 |
|---|---|---|
| **預期 AUROC 損失** | ⬇ 越低越好 | 最重要的數字。**0.003 幾乎沒差；0.03 是常用容忍線；到 0.09 就是不能信** |
| **綠燈／紅燈** | 綠 = 可信 | 紅燈的正確處置是**標記為不可用**，不是硬補完照常判讀 |
| **實際缺失率** | — 中性 | 確認滑桿真的生效，應該幾乎等於你拉的數字 |
| **最長缺口** | ⬇ 越短越好 | 超過 **0.2 秒**大概吃掉一次心跳；超過 **0.84 秒**是一整個心動週期消失 |
| **缺口段數** | — **中間最糟** | 不是越少越好也不是越多越好。切成 5–20 段時傷害最大 |
| **模型的信心 %** | ⚠️ **陷阱** | **不要用它判斷資料好不好。** 模型錯得離譜時一樣可以有 97% 的信心 |

#### 「可容忍的 AUROC 損失」該設多少

**這個值不該由模型開發者決定，該由臨床風險決定**，所以做成可調的。

- **0.01（嚴格）**：漏診代價極高、或這是唯一的判讀依據時。
- **0.03（預設）**：門檻表用的值。模型是輔助、還有醫師覆核時的合理起點。
- **0.05 以上**：只建議用在初篩、後面還有其他關卡的情境。

拉動它，上方綠燈紅燈的界線會跟著動——**那條線是人設的，不是資料算出來的**。

#### 「預期損失」是怎麼算出來的（重要但容易誤會）

它**不是**這一筆算出來的，是從 **213 次評估**（2158 筆的平均）依你目前的設定內插而來。
所以它是**族群層級的期望損失**，代表「這種缺法、這個缺失率下，模型平均會退步多少」，
不是「這一筆的誤差有多大」。右邊那張圖上的 ★ 就是你現在的設定落在哪。
""")

# ─────────────────────────────────────────────── 我們量到什麼
with tab3:
    st.markdown("""
#### 主結果：真正的分水嶺不是「連續 vs 散布」，是「缺失有沒有挑地方發生」

缺失率 20%、線性內插時，AUROC 相對乾淨基線（0.8894）的損失：

| 缺失型態 | 損失 | 相對隨機散布 |
|---|---|---|
| 隨機散布 scatter（MCAR） | 0.0025 | 1× |
| 連續大段 burst（**仍是** MCAR） | 0.0085 | 3.4× |
| **訊號相關 MNAR** | **0.0937** | **37×** |

缺 40% 時，MNAR 讓 AUROC 從 0.889 掉到 **0.746**。

> 這推翻了我自己的原始假設。我本來以為主軸是「連續 vs 散布」——
> 實測是連續型確實比散布型糟（3.4 倍），**但兩者的絕對傷害都很小**。

#### 資料品質門檻表（容忍上限 0.03）

| 缺失型態 | 最大可接受比例 | 建議補值 | 超過時 |
|---|---|---|---|
| 隨機散布 | 40% | 前值填補或線性內插 | 仍可用，改用內插 |
| 連續大段 | 40% | **零填補** | 不要用內插，填中性值 |
| 訊號相關 | **10%** | **零填補** | **標記為不可用** |

#### 另外兩個發現

**最佳補值方法會翻轉。** 零填補在散布型最差（0.038），在連續型與 MNAR 卻最好。
機制可驗證：散布型下零填補在每個缺失點製造一次高頻人工不連續，
實測某一筆訊號 15–40 Hz 相對功率由 **0.128 升到 0.331**，而該頻帶正是模型第二重要的特徵。

**傷害對碎裂程度非單調。** 同樣缺 20%，切成 0.1–0.4 秒的中等缺口最糟，兩端都較輕。
**這個現象目前無法解釋**——三個假設（被破壞的心搏比例、受影響範圍 × 重建誤差、特徵空間位移量）
全部被自己的資料否證。

---

#### 你可以自己驗證

上面每一個數字都來自 repo 裡可重現的實驗紀錄。下面是 demo 實際用的那張傷害表，
你拉滑桿時看到的「預期損失」就是從這裡內插出來的：
""")
    st.dataframe(
        harm.assign(**{"AUROC 損失": (-harm.d).round(4)})
            .rename(columns={"pattern": "缺失型態", "rate": "缺失率",
                             "imputer": "補值方法", "auroc": "AUROC"})
            [["缺失型態", "缺失率", "補值方法", "AUROC", "AUROC 損失"]],
        use_container_width=True, height=260)
    st.caption("原始資料：results/m5_summary_table.csv · 完整實驗紀錄：results/runs.parquet")

# ─────────────────────────────────────────────── 別誤讀
with tab4:
    st.markdown("""
下面四句話聽起來都很順，但都是錯的。

**✕「損失小，就代表這個補值方法補得好。」**
訊號補得像不代表模型判讀得對。訊號層的重建誤差排序與下游 AUROC 的排序，
**翻轉點不在同一個位置**——這代表「補值品質」這個常用指標，跟真正重要的下游表現是脫節的。

**✕「模型信心高，所以這筆可以信。」**
反例就在這個頁面上：97% 的信心配上完全錯誤的答案。
信心來自特徵落在決策邊界的哪一側，跟輸入品質無關。
**要判斷能不能信，需要一個獨立於模型之外的品質分數。**

**✕「守門之後 AUROC 比沒缺失還高，所以守門讓模型變強了。」**
那是**選樣效應**——被留下來的樣本本來就比較好判。
守門不會讓模型變強，它只是讓模型不要回答它答不好的那些。

**✕「這個 demo 做了 MCAR、MAR、MNAR 三種機制。」**
統計上錯了。**隨機散布與連續大段同屬 MCAR**，差別只在空間結構，只有第三種是 MNAR。
正確的說法反而更有力：**在同一個機制之下，光是缺失的結構不同，下游傷害就差 37 倍
——這件事單看機制分類是看不出來的。**

---

#### 這個 demo 的限制

- 只用 **40 筆**預存樣本，不是整個測試集；「預期損失」才是全體 2158 筆的統計。
- 使用 **100 Hz** 降取樣版本，高頻細節已先損失一部分，絕對數值可能低估。
- **假設遮罩是已知的**。真實資料裡缺失以異常值或空值呈現，
  要先判斷哪些不可用——**偵測是補值之前的另一個問題**。

完整方法、決策紀錄與被否證的假設：見 repo 的 `README.md`、`notes/process_and_decisions.md`
與 `demo/GUIDE.md`。
""")
