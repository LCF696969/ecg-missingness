"""M6 — 最終圖三（延長版碎裂曲線）、圖四（守門曲線）與門檻表。

圖三改用延長到 k=200 的資料：k=200 時每段缺口正好 1 點，**定義上就等於散布型**，
所以曲線右端必須碰到散布型的參考線——這是注入器邏輯的收斂驗證，不是裝飾。

圖四改用「高傷害混合」情境（m5c）：M5b 顯示在低缺失率為主的混合下守門幾乎沒有收益，
因為沒有傷害可守。守門機制的價值必須在有傷害的情境下衡量。

產出：
  figures/m5_fig3_fragmentation.png   （覆蓋舊版）
  figures/m5_fig4_coverage.png        （覆蓋舊版）
  results/m6_threshold_table.csv
  results/m6_threshold_table.md
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
from sklearn.metrics import roc_auc_score  # noqa: E402

import config  # noqa: E402

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8a85", "#e3e3df"
IMP_C = {"zero": "#4a3aa7", "ffill": "#eda100", "linear": "#e34948"}
IMP_M = {"zero": "o", "ffill": "s", "linear": "D"}
PATTERN_LABEL = {"scatter": "Scattered (MCAR)", "burst": "Burst (MCAR, structured)",
                 "mnar": "Signal-dependent (MNAR)"}
SQI_C = {"missing_frac": "#2a78d6", "flat_frac": "#eb6834", "max_gap": "#1baf7a"}
SQI_M = {"missing_frac": "o", "flat_frac": "s", "max_gap": "^"}
SQI_LABEL = {"missing_frac": "missing fraction", "flat_frac": "flat-segment fraction",
             "max_gap": "longest gap"}


def style(ax):
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=0)


def main() -> int:
    R = config.RESULTS_DIR
    runs = pd.read_parquet(R / "runs.parquet")
    ext = pd.read_parquet(R / "m5b_ksweep_extended.parquet")
    clean = float(runs.auroc_clean.iloc[0])

    # ================================================= 圖三（延長版）
    ks = pd.concat([runs[runs.group == "ksweep"][["k", "imputer", "auroc", "delta_auroc"]],
                    ext[["k", "imputer", "auroc", "delta_auroc"]]])
    g = ks.groupby(["imputer", "k"], as_index=False).agg(
        auroc=("auroc", "mean"), sd=("auroc", "std"))
    scat = (runs[(runs.group == "main") & (runs.rate == 0.20) &
                 (runs.pattern == "scatter")]
            .groupby("imputer").auroc.mean())

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    style(ax)
    ax.axhline(clean, color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=1)
    ax.annotate(f"no missing data  {clean:.3f}", xy=(1, clean), xytext=(2, 5),
                textcoords="offset points", color=INK2, fontsize=8.5)
    dy = {"zero": -14, "ffill": 8, "linear": -4}
    for imp in ["zero", "ffill", "linear"]:
        s = g[g.imputer == imp].sort_values("k")
        ax.fill_between(s.k, s.auroc - s.sd, s.auroc + s.sd,
                        color=IMP_C[imp], alpha=0.15, lw=0, zorder=2)
        ax.plot(s.k, s.auroc, color=IMP_C[imp], lw=2.0, marker=IMP_M[imp], ms=7,
                mec="white", mew=1.4, zorder=3, label=imp)
        ax.scatter([200], [scat[imp]], marker="*", s=160, color=IMP_C[imp],
                   edgecolor="white", linewidth=1.2, zorder=4)
        ax.annotate(f"scatter {scat[imp]:.3f}", xy=(200, scat[imp]),
                    xytext=(-4, dy[imp]), textcoords="offset points", ha="right",
                    color=INK, fontsize=8.5)
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 5, 10, 20, 50, 100, 200])
    ax.set_xticklabels(["1\n2.0s", "2\n1.0s", "5\n0.4s", "10\n0.2s", "20\n0.1s",
                        "50\n0.04s", "100\n0.02s", "200\n0.01s"], fontsize=8.5)
    ax.set_xlabel("k = number of gaps   (below: length of each gap)",
                  color=INK2, fontsize=10)
    ax.set_ylabel("AUROC", color=INK2, fontsize=10)
    ax.set_title("The same 20% missing hurts most when broken into medium-sized gaps",
                 color=INK, fontsize=12.5, loc="left", pad=26)
    ax.text(0, 1.035, "★ = scattered missingness at the same rate. At k=200 each gap is 1 "
                      "sample, so burst IS scatter — the lines must meet the stars.",
            transform=ax.transAxes, color=INK2, fontsize=8.5)
    ax.legend(frameon=False, fontsize=9.5, labelcolor=INK, loc="lower left", ncol=3)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "m5_fig3_fragmentation.png", dpi=150,
                facecolor="#fcfcfb")
    plt.close(fig)
    print("圖三（延長版）完成")

    # ================================================= 圖四（守門，高傷害情境）
    hd = pd.read_parquet(R / "m5c_gate_highdamage.parquet")
    covs = np.arange(0.15, 1.001, 0.05)
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    style(ax)
    ax.axhline(clean, color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=1)
    ax.annotate(f"no missing data  {clean:.3f}", xy=(covs[0] * 100, clean),
                xytext=(2, 5), textcoords="offset points", color=INK2, fontsize=8.5)

    def curve(s, key):
        order = (np.random.default_rng(0).permutation(len(s)) if key == "random"
                 else np.argsort(s[key].values, kind="stable"))
        out = []
        for cv in covs:
            idx = order[:int(len(s) * cv)]
            yy, pp = s.label.values[idx], s.prob.values[idx]
            out.append(roc_auc_score(yy, pp) if len(np.unique(yy)) > 1 else np.nan)
        return np.array(out)

    for key in ["missing_frac", "flat_frac", "max_gap"]:
        ys = np.stack([curve(hd[hd.seed == sd].reset_index(drop=True), key)
                       for sd in [0, 1, 2]])
        m, e = ys.mean(0), ys.std(0)
        ax.fill_between(covs * 100, m - e, m + e, color=SQI_C[key], alpha=0.15, lw=0, zorder=2)
        ax.plot(covs * 100, m, color=SQI_C[key], lw=2.0, marker=SQI_M[key], ms=6,
                mec="white", mew=1.2, zorder=3, label=SQI_LABEL[key])
    ysr = np.stack([curve(hd[hd.seed == sd].reset_index(drop=True), "random")
                    for sd in [0, 1, 2]])
    ax.plot(covs * 100, ysr.mean(0), color=MUTED, lw=1.8, ls=(0, (5, 3)), zorder=3,
            label="random order (control)")
    ax.set_xlabel("coverage — % of records the model is allowed to answer",
                  color=INK2, fontsize=10)
    ax.set_ylabel("AUROC on answered records", color=INK2, fontsize=10)
    ax.set_title("A label-free quality score that knows when not to answer",
                 color=INK, fontsize=12.5, loc="left", pad=26)
    ax.text(0, 1.035, "high-damage mixture: per-record rate 0–40%, 60% signal-dependent · "
                      "linear interpolation · 3 seeds",
            transform=ax.transAxes, color=INK2, fontsize=8.5)
    ax.legend(frameon=False, fontsize=9.5, labelcolor=INK, loc="lower left")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "m5_fig4_coverage.png", dpi=150, facecolor="#fcfcfb")
    plt.close(fig)
    print("圖四（守門）完成")

    # ================================================= 門檻表
    main_df = runs[runs.group == "main"]
    agg = main_df.groupby(["pattern", "rate", "imputer"], as_index=False).agg(
        d=("delta_auroc", "mean"), lo=("delta_ci_lo", "mean"), hi=("delta_ci_hi", "mean"),
        auroc=("auroc", "mean"))
    rows = []
    for tol in [0.01, 0.03]:
        for pat in ["scatter", "burst", "mnar"]:
            s = agg[agg.pattern == pat]
            best_rate, best_imp, best_auroc = 0.0, None, None
            for rate in sorted(s.rate.unique()):
                sr = s[s.rate == rate]
                b = sr.loc[sr.d.idxmax()]         # 該缺失率下傷害最小的補值法
                if -b.d < tol:
                    best_rate, best_imp, best_auroc = rate, b.imputer, b.auroc
            # 全區間最佳補值法（以最高缺失率下的表現決定建議）
            top = s[s.rate == s.rate.max()]
            rec_imp = top.loc[top.d.idxmax()].imputer
            rows.append({
                "tolerance_dAUROC": tol, "pattern": pat,
                "max_acceptable_rate": best_rate,
                "imputer_at_threshold": best_imp,
                "auroc_at_threshold": best_auroc,
                "recommended_imputer": rec_imp,
            })
    tbl = pd.DataFrame(rows)
    tbl.to_csv(R / "m6_threshold_table.csv", index=False)

    lines = ["# 資料品質門檻表", "",
             f"基準：無缺失時 AUROC = {clean:.4f}（PTB-XL fold 10，2158 筆，固定模型）", "",
             "「最大可接受缺失率」= 在該型態下，效能損失仍小於容忍上限的最高缺失率；",
             "補值方法取該缺失率下傷害最小者。", ""]
    for tol in [0.01, 0.03]:
        lines += [f"## 容忍上限 ΔAUROC < {tol:.2f}", "",
                  "| 缺失型態 | 最大可接受比例 | 該點的補值方法 | 該點 AUROC | 高缺失率下建議的補值 |",
                  "|---|---|---|---|---|"]
        for pat in ["scatter", "burst", "mnar"]:
            r = tbl[(tbl.tolerance_dAUROC == tol) & (tbl.pattern == pat)].iloc[0]
            rate_s = f"{r.max_acceptable_rate:.1%}" if r.max_acceptable_rate > 0 else "**低於 2.5%**"
            au = f"{r.auroc_at_threshold:.4f}" if pd.notna(r.auroc_at_threshold) else "—"
            imp = r.imputer_at_threshold if isinstance(r.imputer_at_threshold, str) else "—"
            lines += [f"| {PATTERN_LABEL[pat]} | {rate_s} | {imp} | {au} | {r.recommended_imputer} |"]
        lines += [""]
    (R / "m6_threshold_table.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
