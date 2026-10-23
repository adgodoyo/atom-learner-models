"""Thesis Figures 10, 11 and 12 (planner). Writes fig10, fig11, fig12 (.pdf/.png) here.

Inputs: results/sim2real_w/real_exam.json for Figure 11; Figures 10 and 12 use the published values
listed in the script, each taken from the result file named beside it (see RESULTS.md).
"""

import json
import os
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = str(pathlib.Path(__file__).resolve().parents[2])
OUT = str(pathlib.Path(__file__).resolve().parent)
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.fontsize": 8.5,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "figure.dpi": 200,
})
BLUE, ORANGE, GREY, GREEN, RED = "#1d4e89", "#b45309", "#8a887f", "#128a5e", "#a33"
RULE = "#e2e0d8"


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, name + ".png"), bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# Figure 10 — where a planner helps, and where it does not.
# =====================================================================
G0 = [  # CodeWorkout, registered headline scenario (start 0, budget 3): point estimates only
    ("trained actor",      -0.148),
    ("greedy EPIG",        -0.128),
    ("difficulty-matched", -0.044),
    ("maximum entropy",    -0.041),
]
G1 = [  # LLM selector fine-tuned on the simulator's pricing; learner-clustered 95% CI
    ("start 3,  budget 3",  -0.0770, -0.1425, -0.0110),
    ("start 3,  budget 8",  -0.0209, -0.0656,  0.0228),
    ("start 3,  budget 16", -0.0400, -0.0686, -0.0114),
    ("start 10, budget 3",  -0.0739, -0.1302, -0.0218),
    ("start 10, budget 8",  -0.0504, -0.0951, -0.0068),
    ("start 10, budget 16", -0.0380, -0.0817,  0.0018),
]
G2 = [("greedy EPIG", 0.0159, 0.0056, 0.0268)]

fig, ax = plt.subplots(figsize=(6.9, 5.6))
y = 0.0
ticks, labels, spans = [], [], {}

y0 = y
for lab, pt in G0:
    ax.plot([pt], [y], marker="D", color=BLUE, ms=6.5, mfc="white", mec=BLUE, mew=1.7, zorder=3)
    ticks.append(y); labels.append(lab); y -= 1
spans[0] = (y0, y + 1); y -= 1.5

y0 = y
for lab, pt, lo, hi in G1:
    col = BLUE if hi < 0 else (RED if lo > 0 else "#5b7fa6")
    ax.errorbar([pt], [y], xerr=[[pt - lo], [hi - pt]], fmt="o", color=col,
                capsize=2.5, lw=1.3, ms=4.6, mfc=col, mec=col, zorder=3)
    ticks.append(y); labels.append(lab); y -= 1
spans[1] = (y0, y + 1); y -= 1.5

y0 = y
for lab, pt, lo, hi in G2:
    ax.errorbar([pt], [y], xerr=[[pt - lo], [hi - pt]], fmt="o", color=RED,
                capsize=2.5, lw=1.3, ms=4.6, mfc=RED, mec=RED, zorder=3)
    ticks.append(y); labels.append(lab); y -= 1
spans[2] = (y0, y + 1)

ax.axvline(0, color="#333", lw=1.1, zorder=2)
ax.set_yticks(ticks); ax.set_yticklabels(labels)
ax.tick_params(axis="y", length=0)
ax.set_xlim(-0.20, 0.20)
bot = spans[2][1]
ax.set_ylim(bot - 2.4, 1.9)

titles = {
    0: "CodeWorkout · heuristics and the trained actor",
    1: "CodeWorkout · language model fine-tuned on the simulator's pricing",
    2: "FoundationalASSIST · the same estimator, the other corpus",
}
for g, (top, _b) in spans.items():
    ax.text(-0.198, top + 0.38, titles[g], fontsize=8.4, fontweight="bold", color="#333", va="bottom")
for g in (0, 1):
    _t, b = spans[g]
    ax.axhline(b - 0.75, color=RULE, lw=1.0, zorder=1)

ax.text(0.0, 2.05, "random", fontsize=8.5, color="#333", ha="center", va="bottom", fontweight="bold")
ax.annotate("", xy=(-0.155, 1.35), xytext=(-0.045, 1.35),
            arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.4))
ax.text(-0.100, 1.48, "better than random", fontsize=8.2, color=BLUE, ha="center", va="bottom")
ax.annotate("", xy=(0.155, 1.35), xytext=(0.045, 1.35),
            arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.4))
ax.text(0.100, 1.48, "worse than random", fontsize=8.2, color=RED, ha="center", va="bottom")

ax.set_xlabel("log-loss relative to a random policy on the same menus (nats)")
ax.text(-0.198, bot - 1.25,
        "Circles carry a learner-clustered 95 per cent interval. Open diamonds are the registered headline scenario,\n"
        "reported as point estimates only.",
        fontsize=7.4, color=GREY, va="top")
fig.tight_layout()
save(fig, "fig10_planner_forest")
print("wrote Figure 10")


# =====================================================================
# Figure 11 — transfer of a policy trained with zero real rows.
# =====================================================================
real_exam = json.load(open(os.path.join(ROOT, "results/sim2real_w/real_exam.json")))
budgets = [3, 5, 10]
series = {}
for start in (3, 10):
    pts, los, his = [], [], []
    for b in budgets:
        cell = real_exam["cells"][f"start{start}|B{b}"]
        sa = cell["policies"]["sim_actor"]
        rn = cell["policies"]["random"]
        pts.append(sa[0] / rn[0])
        los.append(sa[1] / rn[2])   # conservative bound from the two marginal intervals
        his.append(sa[2] / rn[1])
    series[start] = (np.array(pts), np.array(los), np.array(his))

fig, ax = plt.subplots(figsize=(6.0, 3.9))
styles = {3: (BLUE, "o", "start 3 (a cold-ish belief)"),
          10: (ORANGE, "s", "start 10 (a warm belief)")}
x = np.array(budgets, dtype=float)
for start in (3, 10):
    pts, _lo, _hi = series[start]
    col, mk, lab = styles[start]
    ax.plot(x, pts, marker=mk, color=col, lw=1.9, ms=7, label=lab,
            mfc="white", mec=col, mew=1.8, zorder=3)

ax.axhline(1.0, color="#333", lw=1.1, ls="--", zorder=2)
ax.text(10.85, 1.008, "random", fontsize=8.3, color="#333", va="bottom", ha="right")
pts10 = series[10][0]
ax.annotate("1.130 \u2014 the one reversal:\nthe warmest belief,\nthe largest budget",
            xy=(9.85, pts10[-1]), xytext=(7.0, 1.21),
            arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.2),
            fontsize=8.2, color=ORANGE, ha="center")

ax.set_xticks(budgets); ax.set_xticklabels([str(b) for b in budgets])
ax.set_xlabel("budget: questions the policy is allowed to ask")
ax.set_ylabel("score relative to a random policy\n(centred-logit MSE ratio; below 1 is better)")
ax.set_xlim(2.4, 10.9)
ax.set_ylim(0.50, 1.34)
ax.legend(loc="lower right")
ax.text(2.5, 1.315,
        "Markers are open because no scenario separates from random: all six intervals overlap.",
        fontsize=7.5, color=GREY, va="top")
fig.tight_layout()
save(fig, "fig11_sim_trained_transfer")
print("wrote Figure 11", {s: list(np.round(series[s][0], 3)) for s in series})


# =====================================================================
# Figure 12 — the simulator teaches a language model to select.
# =====================================================================
tier3_headroom = [
    ("start 3, B3",   0.129, 0.252),
    ("start 3, B8",  -0.283, 0.395),
    ("start 3, B16", -0.162, 0.470),
    ("start 10, B3", -0.120, 0.419),
    ("start 10, B8", -0.008, 0.641),
    ("start 10, B16", 0.342, 0.743),
]
fig, ax = plt.subplots(figsize=(5.8, 4.2))
for lab, zs, tr in tier3_headroom:
    ax.plot([0, 1], [zs, tr], "-", color=GREY, lw=1.1, alpha=0.8, zorder=1)
    ax.plot(0, zs, "o", color=RED, ms=6, zorder=2)
    ax.plot(1, tr, "o", color=BLUE, ms=6, zorder=2)
    ax.text(1.06, tr, lab, fontsize=7.4, color="#444", va="center")
ax.axhline(0, color="#333", lw=1.0, ls="--")
ax.text(-0.12, 0.015, "random", fontsize=8.2, color="#333", ha="left", va="bottom")
ax.set_xlim(-0.18, 1.55)
ax.set_ylim(-0.42, 0.85)
ax.set_xticks([0, 1])
ax.set_xticklabels(["zero-shot\n(same backbone, untrained)",
                    "fine-tuned on the\nsimulator's own pricing"])
ax.set_ylabel("share of the random$\\to$oracle gap closed\n(0 = random, 1 = the privileged oracle)")
fig.tight_layout()
save(fig, "fig12_llm_selector")
print("wrote Figure 12")
