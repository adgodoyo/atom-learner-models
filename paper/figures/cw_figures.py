"""Thesis Figures 6 and 9 (CodeWorkout). Writes fig06 and fig09 (.pdf/.png) here.

Inputs: data/fig06_cw_maturity_auc.csv (from scripts/cw_fig06_maturity_data.py) and
data/fig09_cw_ceiling_bins.csv (from scripts/cw_classmate_ceiling.py); both are bucket-level aggregates.
"""
import os
import pathlib

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = str(HERE)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.fontsize": 8.5,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "figure.dpi": 200,
})
BLUE, ORANGE, GREY = "#1d4e89", "#b45309", "#8a887f"


# ---------------------------------------------------------------- Figure 6
m = pd.read_csv(DATA / "fig06_cw_maturity_auc.csv")
ref = float(m.loc[m["bucket"] == "all", "difficulty_only_auc"].iloc[0])
m = m[m["bucket"] != "all"].reset_index(drop=True)
x = np.arange(len(m))
auc, lo, hi = m["auc"].to_numpy(), m["lo"].to_numpy(), m["hi"].to_numpy()
fig, ax = plt.subplots(figsize=(6.4, 3.4))
ax.errorbar(x, auc, yerr=[auc - lo, hi - auc], fmt="o-", color=BLUE, capsize=3, lw=1.8, ms=6)
ax.axhline(0.5, color="k", ls="--", lw=0.9)
ax.text(4.35, 0.505, "chance", fontsize=8, ha="right")
ax.axhline(ref, color=GREY, ls=":", lw=1.2)
ax.text(4.35, ref + 0.008, "knowing only how hard the problem is", fontsize=8, color="#4a4a4a",
        ha="right", va="bottom")
ax.annotate("first problem of the course:\nthe model knows nothing\nabout this learner",
            xy=(0.07, 0.44), xytext=(0.42, 0.345), fontsize=8, color="#333",
            arrowprops=dict(arrowstyle="->", color="#333", lw=0.9))
for i2 in range(len(m)):
    ax.text(i2, hi[i2] + 0.014, f"{auc[i2]:.3f}", ha="center", fontsize=8.5, fontweight="bold", color=BLUE)
ax.set_xticks(x); ax.set_xticklabels(m["bucket"].astype(str))
ax.set_xlim(-0.4, 4.4); ax.set_ylim(0.30, 0.95)
ax.set_xlabel("problems the learner had already worked through")
ax.set_ylabel("AUC: will this submission pass\nevery test on the first try?")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig06_cw_belief_maturity.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "fig06_cw_belief_maturity.png"), bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- Figure 9
b = pd.read_csv(DATA / "fig09_cw_ceiling_bins.csv")
STRATA = [(0, 0, "cold\nstart"), (1, 8, "1–8"), (9, 16, "9–16"), (17, 36, "17–36"), (37, 84, "37+")]
gapv, gap_lo, gap_hi = (b[c].to_numpy() for c in ("sem_gap_atom_minus_peeravg", "sem_gap_lo", "sem_gap_hi"))
atom90, atom90_lo, atom90_hi = (b[c].to_numpy() for c in ("char_pct_ge90_atom", "char_pct_ge90_atom_lo",
                                                         "char_pct_ge90_atom_hi"))
med90, med90_lo, med90_hi = (b[c].to_numpy() for c in ("char_pct_ge90_medoid", "char_pct_ge90_medoid_lo",
                                                      "char_pct_ge90_medoid_hi"))
labels = [s[2] for s in STRATA]
x = np.arange(len(STRATA), dtype=float)

fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.4))

# ---------------- (a) advantage over the population average ----------------
ax = axes[0]
ax.fill_between(x, gap_lo, gap_hi, color=BLUE, alpha=0.15, lw=0)
ax.plot(x, gapv, "o-", color=BLUE, lw=1.9, ms=5.5, zorder=3)
ax.axhline(0, color="#333", lw=1.0)
ax.text(0.06, gapv[0] + 0.016, f"{gapv[0]:.3f}", fontsize=8.2, color=BLUE, fontweight="bold")
ax.text(len(x) - 1.30, gapv[-1] + 0.012, f"{gapv[-1]:.3f}", fontsize=8.2, color=BLUE, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_xlim(-0.25, len(x) - 0.75)
ax.set_ylim(-0.02, 0.22)
ax.set_ylabel("ATOM's advantage over a random\nclassmate's code (semantic similarity)")
ax.set_xlabel("episodes in the belief when this attempt was made")
ax.set_title("(a) Advantage over the population average,\nmeaning-level, composition-controlled")

# ---------------- (b) near-perfect character matches ----------------
ax = axes[1]
ax.fill_between(x, atom90_lo, atom90_hi, color=BLUE, alpha=0.15, lw=0)
ax.fill_between(x, med90_lo, med90_hi, color=ORANGE, alpha=0.15, lw=0)
ax.plot(x, atom90, "o-", color=BLUE, lw=1.9, ms=5.5, label="ATOM generated", zorder=3)
ax.plot(x, med90, "s-", color=ORANGE, lw=1.9, ms=5.5, label="medoid (learner-blind ceiling)", zorder=3)
ax.axvline(2.5, color=GREY, ls=":", lw=1.1)
ax.text(2.42, 19.0, "crossover", ha="right", fontsize=8, color=GREY)
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_xlim(-0.25, len(x) - 0.75)
ax.set_ylim(0, 33)
ax.set_ylabel("share of generations $\\geq$90%\nsimilar to the real submission")
ax.set_xlabel("episodes in the belief when this attempt was made")
ax.set_title("(b) Near-perfect matches, character space")
ax.legend(loc="upper left")

fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig09_cw_ceiling_by_maturity.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "fig09_cw_ceiling_by_maturity.png"), bbox_inches="tight")
plt.close(fig)
print("wrote fig06 and fig09")
