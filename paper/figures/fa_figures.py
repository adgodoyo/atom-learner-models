"""Thesis Figures 5, 7 and 8 (FoundationalASSIST). Writes fig05, fig07, fig08 (.pdf/.png) here.

Inputs: data/fig05_fa_auc_by_history.csv (from fig05_data.py) and
results/fa_sim3_kt_panel/panel.json. Figure 8's eight effects are the published values listed below.
"""
import os
import pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = pathlib.Path(__file__).resolve().parents[2]
DATA = os.path.join(pathlib.Path(__file__).resolve().parent, "data")
OUT = str(pathlib.Path(__file__).resolve().parent)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.fontsize": 8.5,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "figure.dpi": 200,
})

BLUE, ORANGE, GREY = "#1f4e79", "#c05621", "#8a8a8a"
LAB = ["0-9", "10-24", "25-49", "50-99", "100-199", "200+"]
x = np.arange(6)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, name + ".png"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- Figure 5
a = pd.read_csv(os.path.join(DATA, "fig05_fa_auc_by_history.csv"))

fig, ax = plt.subplots(figsize=(6.4, 3.4))
ax.errorbar(x, a.auc_own, yerr=[a.auc_own - a.auc_own_lo, a.auc_own_hi - a.auc_own],
            fmt="o-", color=BLUE, capsize=3, lw=1.8, ms=6, label="this learner's belief")
ax.errorbar(x, a.auc_shuffled,
            yerr=[a.auc_shuffled - a.auc_shuffled_lo, a.auc_shuffled_hi - a.auc_shuffled],
            fmt="s--", color=ORANGE, capsize=3, lw=1.6, ms=5,
            label="another learner's belief")
ax.axhspan(0.512, 0.559, color=GREY, alpha=0.25, lw=0)
ax.axhline(0.5, color="k", ls="--", lw=0.8)
ax.text(-0.40, 0.4935, "grey band: four frontier models (70-120B) reading the whole history as text;\n"
        "their own published curve is flat across this axis", fontsize=7.5, color="#4a4a4a",
        ha="left", va="top")
for i in (0, 5):
    d = a.auc_own[i] - a.auc_shuffled[i]
    ax.annotate("", xy=(i + 0.17, a.auc_own[i]), xytext=(i + 0.17, a.auc_shuffled[i]),
                arrowprops=dict(arrowstyle="<->", color="k", lw=0.9))
    ax.text(i + 0.25, (a.auc_own[i] + a.auc_shuffled[i]) / 2, f"+{d:.3f}",
            fontsize=8.5, va="center", fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(LAB)
ax.set_xlim(-0.45, 5.75)
ax.set_ylim(0.440, 0.885)
ax.set_xlabel("events the belief has consumed before the question is asked")
ax.set_ylabel("AUC: will this learner answer correctly?")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
fig.tight_layout()
save(fig, "fig05_fa_locating_the_learner")


# ---------------------------------------------------------------- Figure 7
import json
_hc = json.load(open(REPO / "results" / "fa_sim3_kt_panel" / "panel.json"))["history_curve"]
hist = pd.DataFrame({"identity_gap_vs_shuffled_u": [r["identity_gap_vs_shuffled_u"][0] for r in _hc],
                     "identity_gap_vs_shuffled_u_ci_lo": [r["identity_gap_vs_shuffled_u"][1] for r in _hc],
                     "identity_gap_vs_shuffled_u_ci_hi": [r["identity_gap_vs_shuffled_u"][2] for r in _hc]})

fig, ax = plt.subplots(figsize=(6.0, 3.1))
g = hist["identity_gap_vs_shuffled_u"]
ax.errorbar(x, g,
            yerr=[g - hist["identity_gap_vs_shuffled_u_ci_lo"],
                  hist["identity_gap_vs_shuffled_u_ci_hi"] - g],
            fmt="o-", color=BLUE, capsize=3, lw=1.8, ms=6)
ax.axhline(0.0, color="k", lw=0.9, ls="--")
ax.axhline(0.016832, color=GREY, ls=":", lw=1.2)
ax.text(0.05, 0.0176, "pooled +0.0168", fontsize=8, color="#4a4a4a")
ax.set_xticks(x); ax.set_xticklabels(LAB)
ax.set_xlabel("events the belief has consumed")
ax.set_ylabel("likelihood gained by using\nthis learner's belief (nats)")
fig.tight_layout()
save(fig, "fig07_fa_identity_gap_maturity")


# ---------------------------------------------------------------- Figure 8
# (effect, CI low, CI high, n): from results/fa_sim3_full_run/summary.json and results/fa_sim3_eval_5e5f/summary.json
rows = [
    ("ranking of who succeeds", 0.1968, 0.1761, 0.2167, 4977, "cont"),
    ("likelihood of the answer", 0.01683, 0.01333, 0.02045, 4977, "cont"),
    ("semantic similarity of the text", 0.0027, -0.0033, 0.0093, 300, "cont"),
    ("character distance to the text", 0.0115, -0.0099, 0.0350, 300, "disc"),
    ("exact answer, open response", 0.0105, 0.0039, 0.0177, 3325, "disc"),
    ("exact answer, multiple choice", -0.0021, -0.0113, 0.0063, 968, "disc"),
    ("exact answer, select all", -0.0063, -0.0164, 0.0032, 633, "disc"),
    ("exact answer, free generation", 0.0033, -0.0300, 0.0370, 300, "disc"),
]
lab = [f"{r[0]}   (n={r[4]:,})" for r in rows]
ratio = [r[1] / ((r[3] - r[2]) / 2) for r in rows]
cols = [BLUE if r[5] == "cont" else ORANGE for r in rows]
y = np.arange(len(rows))[::-1]

fig, ax = plt.subplots(figsize=(7.0, 3.5))
bars = ax.barh(y, ratio, color=cols, height=0.62)
ax.axvline(0, color="k", lw=0.9)
ax.axvline(1, color="k", ls="--", lw=1.1)
ax.text(1.15, y[0] + 0.28, "detection threshold", fontsize=8)
ax.set_yticks(y); ax.set_yticklabels(lab)
ax.set_xlabel("identity effect, in units of its own 95% interval half-width")
ax.set_xlim(-1.5, 10.5)
ax.set_ylim(y[-1] - 1.25, y[0] + 0.75)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=BLUE, label="read off the model's probabilities"),
                   Patch(color=ORANGE, label="read off the tokens it emits")],
          loc="lower right", fontsize=8.5)
fig.tight_layout()
save(fig, "fig08_fa_identity_by_readout")

print("wrote 3 figures; gap cold %.4f deep %.4f"
      % (a.auc_own[0] - a.auc_shuffled[0], a.auc_own[5] - a.auc_shuffled[5]))
