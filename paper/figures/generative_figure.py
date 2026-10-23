"""Thesis Figure 13 (writing the question). Writes fig13 (.pdf/.png) here.

Input: results/ga_1_tier4_rungc_panel/summary.json.
"""

import json
import os
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = str(pathlib.Path(__file__).resolve().parents[2])
SRC = os.path.join(ROOT, "results/ga_1_tier4_rungc_panel/summary.json")
OUT = str(pathlib.Path(__file__).resolve().parent)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.fontsize": 8,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "figure.dpi": 200,
})
BLUE, ORANGE, GREY, GREEN, RED = "#1d4e89", "#b45309", "#8a887f", "#128a5e", "#a33"

P = json.load(open(SRC))["panel"]
zs, tu = P["zero_shot|history"], P["tuned|history"]

# ---------------- panel (a): what changed about the questions ----------------
rows = [
    ("on the manifold of real problems",      zs["manifold_support_at_real_p95"],        tu["manifold_support_at_real_p95"],        True),
    ("proposes something already seen",       zs["nearest_already_in_history_fraction"], tu["nearest_already_in_history_fraction"], False),
    ("lands nowhere near a real problem",     zs["projection_miss_rate"],                tu["projection_miss_rate"],                False),
    ("unique generations",                    zs["unique_normalised_fraction"],          tu["unique_normalised_fraction"],          True),
    ("verbatim copies of a real problem",     zs["exact_pool_match_fraction"],           tu["exact_pool_match_fraction"],           False),
]

fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5), gridspec_kw={"width_ratios": [1.18, 1]})
ax = axes[0]
y = np.arange(len(rows))[::-1]
for yi, (lab, a, b, higher_better) in zip(y, rows):
    good = (b > a) if higher_better else (b < a)
    col = GREEN if good else ORANGE
    ax.annotate("", xy=(b, yi), xytext=(a, yi),
                arrowprops=dict(arrowstyle="-|>", color=col, lw=2.0, shrinkA=3, shrinkB=3))
    ax.plot(a, yi, "o", ms=6, color="white", mec=GREY, mew=1.6, zorder=3)
    ax.plot(b, yi, "o", ms=7, color=col, mec="white", mew=1.0, zorder=4)
    ax.text(b + (0.035 if b >= a else -0.035), yi + 0.24, f"{b:.2f}",
            fontsize=8, color=col, fontweight="bold",
            ha="left" if b >= a else "right")
ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows])
ax.tick_params(axis="y", length=0)
ax.set_xlim(-0.08, 1.16)
ax.set_ylim(-0.55, len(rows) - 0.25)
ax.set_xlabel("fraction of the 320 generated questions")
ax.set_title("(a) What fine-tuning changed about the questions")


# ---------------- panel (b): what it bought in value ----------------
ax = axes[1]
inst = [
    ("what the value-of-information\nscorer thinks of them",
     zs["epig_percentile_secondary"], tu["epig_percentile_secondary"]),
    ("what they are actually worth\n(realised diagnostic value)",
     zs["realised_r_percentile_PRIMARY"], tu["realised_r_percentile_PRIMARY"]),
]
ypos = [1.0, 0.0]
for yi, (lab, a, b) in zip(ypos, inst):
    for (v, lo, hi), off, col, mfc in [(a, 0.16, GREY, "white"), (b, -0.16, BLUE, BLUE)]:
        ax.errorbar([v], [yi + off], xerr=[[v - lo], [hi - v]], fmt="o",
                    color=col, mfc=mfc, mec=col, ms=6.5, lw=1.5, capsize=3, zorder=3)

ax.axvline(0.50, color="#333", lw=1.2, ls="--", zorder=2)
ax.text(0.507, -0.58, "a random pick from the menu", fontsize=8, color="#333",
        va="bottom", ha="left")
ax.set_yticks(ypos)
ax.set_yticklabels([i[0] for i in inst], fontsize=8.2)
ax.tick_params(axis="y", length=0)
ax.set_ylim(-0.68, 1.45)
ax.set_xlim(0.15, 0.74)
ax.set_xlabel("percentile within the state's own menu")
ax.set_title("(b) What that bought")

fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig13_generative_tier.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "fig13_generative_tier.png"), bbox_inches="tight")
plt.close(fig)
print("wrote Figure 13")
for lab, a, b, _ in rows:
    print(f"  {lab:38s} {a:.3f} -> {b:.3f}")
for lab, a, b in inst:
    print(f"  {lab.splitlines()[0]:38s} {a[0]:.3f} -> {b[0]:.3f}")
