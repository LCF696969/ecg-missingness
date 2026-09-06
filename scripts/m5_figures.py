"""M5 — 三張核心圖 + 覆蓋率-效能曲線。

與實驗分離：讀 results/runs.parquet 就能重畫，不必重跑 180 次評估。

配色經過色盲安全驗證（OKLab ΔE，all-pairs）：
  缺失型態 blue/orange/aqua  — worst CVD ΔE 9.2、normal-vision ΔE 24.0
  補值方法 violet/yellow/red — worst CVD ΔE 15.3、normal-vision ΔE 20.8
兩組都另外用**不同標記形狀**做次要編碼，因此識別不只依賴顏色。
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as spstats  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

import config  # noqa: E402

# --- 設計 tokens ---
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8a85"
GRID = "#e3e3df"
PATTERN_C = {"scatter": "#2a78d6", "burst": "#eb6834", "mnar": "#1baf7a"}
PATTERN_M = {"scatter": "o", "burst": "s", "mnar": "^"}
PATTERN_LABEL = {"scatter": "Scattered (MCAR)", "burst": "Burst (MCAR, structured)",
                 "mnar": "Signal-dependent (MNAR)"}
IMP_C = {"zero": "#4a3aa7", "ffill": "#eda100", "linear": "#e34948"}
IMP_M = {"zero": "o", "ffill": "s", "linear": "D"}


def style(ax):
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=0)


def agg(df, keys):
    """對注入 seed 取平均，並把配對 bootstrap 的信賴區間一起帶出來。"""
    g = df.groupby(keys, as_index=False).agg(
        auroc=("auroc", "mean"), auroc_sd=("auroc", "std"),
        auprc=("auprc", "mean"),
        d=("delta_auroc", "mean"),
        d_lo=("delta_ci_lo", "mean"), d_hi=("delta_ci_hi", "mean"))
    return g


def main() -> int:
    runs = pd.read_parquet(config.RESULTS_DIR / "runs.parquet")
    clean = float(runs.auroc_clean.iloc[0])
    print(f"乾淨基線 AUROC = {clean:.4f}；共 {len(runs)} 次評估")

    main_df = runs[runs.group == "main"]
    ks_df = runs[runs.group == "ksweep"]

    # =================================================== 圖一：主圖
    g = agg(main_df[main_df.imputer == "linear"], ["pattern", "rate"])
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    style(ax)
    ax.axhline(clean, color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=1)
    ax.annotate(f"no missing data  {clean:.3f}", xy=(0.026, clean), xytext=(0, 5),
                textcoords="offset points", color=INK2, fontsize=8.5, va="bottom")
    label_dy = {"scatter": 7, "burst": -11, "mnar": -3}
    for pat in ["scatter", "burst", "mnar"]:
        s = g[g.pattern == pat].sort_values("rate")
        x = s.rate * 100
        ax.fill_between(x, clean + s.d_lo, clean + s.d_hi,
                        color=PATTERN_C[pat], alpha=0.16, lw=0, zorder=2)
        ax.plot(x, s.auroc, color=PATTERN_C[pat], lw=2.0, marker=PATTERN_M[pat],
                ms=7, mec="white", mew=1.4, zorder=3, label=PATTERN_LABEL[pat])
        ax.annotate(f"{s.auroc.iloc[-1]:.3f}", xy=(x.iloc[-1], s.auroc.iloc[-1]),
                    xytext=(8, label_dy[pat]), textcoords="offset points",
                    color=INK, fontsize=9, fontweight="bold")
    ax.set_xlabel("missing rate (%)", color=INK2, fontsize=10)
    ax.set_ylabel("AUROC", color=INK2, fontsize=10)
    ax.set_title("Same missing rate, different structure — different harm",
                 color=INK, fontsize=12.5, loc="left", pad=26)
    ax.text(0, 1.035, "imputation: linear interpolation · band: paired bootstrap 95% CI · "
                      "fixed model, 3 injection seeds",
            transform=ax.transAxes, color=INK2, fontsize=8.5)
    ax.set_xlim(0, 46)
    ax.legend(frameon=False, fontsize=9.5, labelcolor=INK, loc="lower left")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "m5_fig1_main.png", dpi=150,
                facecolor="#fcfcfb")
    plt.close(fig)
    print("圖一完成")

    # =================================================== 圖二：補值 × 型態
    r = 0.20
    g2 = agg(main_df[main_df.rate == r], ["pattern", "imputer"])
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    style(ax)
    pats = ["scatter", "burst", "mnar"]
    imps = ["zero", "ffill", "linear"]
    w, gap = 0.26, 0.012
    for j, imp in enumerate(imps):
        xs, ys, es = [], [], []
        for i, pat in enumerate(pats):
            row = g2[(g2.pattern == pat) & (g2.imputer == imp)].iloc[0]
            xs.append(i + (j - 1) * (w + gap)); ys.append(row.d); es.append(row.auroc)
        ax.bar(xs, ys, width=w, color=IMP_C[imp], label=imp, zorder=3,
               edgecolor="#fcfcfb", linewidth=1.5)
        for x, y, a in zip(xs, ys, es):
            ax.annotate(f"{a:.3f}", xy=(x, y), xytext=(0, -12 if y < 0 else 4),
                        textcoords="offset points", ha="center",
                        color=INK, fontsize=8.5)
    ax.axhline(0, color=INK2, lw=1.0, zorder=4)
    ax.set_xticks(range(len(pats)))
    ax.set_xticklabels([PATTERN_LABEL[p] for p in pats], color=INK, fontsize=9.5)
    ax.set_ylabel("ΔAUROC vs no missing data", color=INK2, fontsize=10)
    ax.set_title("Does the best imputation method depend on the structure?",
                 color=INK, fontsize=12.5, loc="left", pad=12)
    ax.text(0, 1.015, f"missing rate {r:.0%} · burst k=2 · labels show absolute AUROC",
            transform=ax.transAxes, color=INK2, fontsize=8.5)
    ax.legend(frameon=False, fontsize=9.5, labelcolor=INK, ncol=3, loc="lower right")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "m5_fig2_imputers.png", dpi=150, facecolor="#fcfcfb")
    plt.close(fig)
    print("圖二完成")

    # =================================================== 圖三：碎裂程度
    g3 = agg(ks_df, ["k", "imputer"])
    scat = agg(main_df[(main_df.rate == 0.20) & (main_df.pattern == "scatter")], ["imputer"])
    rr = pd.read_csv(config.RESULTS_DIR / "m3_rr_distribution.csv").iloc[0]
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    style(ax)
    ax.axhline(clean, color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=1)
    ax.annotate(f"no missing data  {clean:.3f}", xy=(1, clean), xytext=(2, 5),
                textcoords="offset points", color=INK2, fontsize=8.5)
    for imp in ["zero", "ffill", "linear"]:
        s = g3[g3.imputer == imp].sort_values("k")
        ax.fill_between(s.k, clean + s.d_lo, clean + s.d_hi,
                        color=IMP_C[imp], alpha=0.14, lw=0, zorder=2)
        ax.plot(s.k, s.auroc, color=IMP_C[imp], lw=2.0, marker=IMP_M[imp], ms=7,
                mec="white", mew=1.4, zorder=3, label=f"{imp} (burst)")
        ref = scat[scat.imputer == imp].auroc.iloc[0]
        ax.axhline(ref, color=IMP_C[imp], lw=1.0, ls=":", alpha=0.75, zorder=1)
        ax.annotate(f"scatter, {imp}", xy=(20, ref), xytext=(4, -3),
                    textcoords="offset points", color=INK2, fontsize=7.5)
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 5, 10, 20])
    ax.set_xticklabels(["1\n2.0 s\n2.4 beats", "2\n1.0 s\n1.2", "5\n0.4 s\n0.5",
                        "10\n0.2 s\n0.24", "20\n0.1 s\n0.12"], fontsize=8.5)
    ax.set_xlabel("k = number of gaps  /  gap length  /  cardiac cycles per gap",
                  color=INK2, fontsize=10)
    ax.set_ylabel("AUROC", color=INK2, fontsize=10)
    ax.set_title("Fragmentation: the same 20% missing, split into k gaps",
                 color=INK, fontsize=12.5, loc="left", pad=12)
    ax.text(0, 1.015, f"dotted lines = scattered missingness at the same rate "
                      f"(the limit burst should approach) · median RR {rr.rr_median:.2f}s",
            transform=ax.transAxes, color=INK2, fontsize=8.5)
    ax.legend(frameon=False, fontsize=9.5, labelcolor=INK, loc="lower right")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "m5_fig3_fragmentation.png", dpi=150, facecolor="#fcfcfb")
    plt.close(fig)
    print("圖三完成")

    # =================================================== 圖四：SQI 覆蓋率-效能
    sqi = pd.read_parquet(config.RESULTS_DIR / "m5_sqi_records.parquet")
    cands = ["max_gap", "missing_frac", "n_gaps", "flat_frac", "rough"]
    sub = sqi[(sqi.rate == 0.20) & (sqi.imputer == "linear")]

    print("\nSQI 候選特徵與「預測誤差」的 Spearman 相關（越高代表越能預測不可信）：")
    corr_rows = []
    for pat in ["scatter", "burst", "mnar"]:
        s = sub[sub.pattern == pat]
        for c in cands:
            rho, pv = spstats.spearmanr(s[c], s.abs_error)
            corr_rows.append({"pattern": pat, "sqi": c, "spearman_rho": rho, "p": pv})
    corr = pd.DataFrame(corr_rows)
    corr.to_csv(config.RESULTS_DIR / "m5_sqi_correlation.csv", index=False)
    print(corr.pivot(index="sqi", columns="pattern", values="spearman_rho").round(3).to_string())

    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    style(ax)
    covs = np.arange(0.2, 1.001, 0.05)
    for pat in ["scatter", "burst", "mnar"]:
        s = sub[sub.pattern == pat].reset_index(drop=True)
        order = np.argsort(s.max_gap.values, kind="stable")   # 品質好（缺口短）在前
        ys = []
        for cv in covs:
            idx = order[:max(30, int(len(s) * cv))]
            yy, pp = s.label.values[idx], s.prob.values[idx]
            ys.append(roc_auc_score(yy, pp) if len(np.unique(yy)) > 1 else np.nan)
        ax.plot(covs * 100, ys, color=PATTERN_C[pat], lw=2.0, marker=PATTERN_M[pat],
                ms=6, mec="white", mew=1.2, zorder=3, label=PATTERN_LABEL[pat])
        ax.annotate(f"{ys[0]:.3f}", xy=(covs[0] * 100, ys[0]), xytext=(-6, 4),
                    textcoords="offset points", ha="right", color=INK, fontsize=8.5)
    ax.axhline(clean, color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=1)
    ax.annotate(f"no missing data  {clean:.3f}", xy=(20, clean), xytext=(2, 5),
                textcoords="offset points", color=INK2, fontsize=8.5)
    ax.set_xlabel("coverage — % of records answered (best signal quality first)",
                  color=INK2, fontsize=10)
    ax.set_ylabel("AUROC on answered records", color=INK2, fontsize=10)
    ax.set_title("A gate that knows when not to answer", color=INK, fontsize=12.5,
                 loc="left", pad=12)
    ax.text(0, 1.015, "quality score = longest gap (label-free) · missing rate 20% · "
                      "linear interpolation",
            transform=ax.transAxes, color=INK2, fontsize=8.5)
    ax.legend(frameon=False, fontsize=9.5, labelcolor=INK, loc="lower left")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "m5_fig4_coverage.png", dpi=150, facecolor="#fcfcfb")
    plt.close(fig)
    print("圖四完成")

    # =================================================== 表格輸出（table view）
    tbl = agg(main_df, ["pattern", "rate", "imputer"])
    tbl.to_csv(config.RESULTS_DIR / "m5_summary_table.csv", index=False)
    agg(ks_df, ["k", "imputer"]).to_csv(config.RESULTS_DIR / "m5_ksweep_table.csv", index=False)
    print("\n表格已輸出（圖的 table view）：m5_summary_table.csv / m5_ksweep_table.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
