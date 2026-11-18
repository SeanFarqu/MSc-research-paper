"""Expected success model: the write-up's two charts, from model.py's report.

python figures.py OUT_DIR
"""

from __future__ import annotations

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = sys.argv[1]
rep = json.load(open(f"{OUT}/report.json"))
sk = rep["skaters_with_scouting_and_more"]
INK, MUTED, GRID = "#1d2433", "#5b6475", "#e6e9ef"
MODEL, DRAFT, BOTH = "#2f6fdb", "#9aa3b2", "#0f9d74"
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.edgecolor": GRID,
        "axes.labelcolor": MUTED,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
    }
)

# 1. calibration: predicted chance of playing against the share who did, by tenth
fig, ax = plt.subplots(figsize=(6.4, 5.2), dpi=160)
pred = [d[0] for d in sk["calibration_deciles"]]
real = [d[1] for d in sk["calibration_deciles"]]
ax.plot([0, 1], [0, 1], color=GRID, lw=1.5, zorder=1)
ax.plot(pred, real, color=MODEL, lw=2, marker="o", ms=8, mec="white", mew=2, zorder=3)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel("Predicted chance of playing an NHL game")
ax.set_ylabel("Share who played within 7 seasons")
ax.set_title("Predicted against actual, by tenth of picks", color=INK, loc="left", fontsize=12)
ax.grid(color=GRID, lw=0.8)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.text(
    0.01,
    0.01,
    f"Skaters drafted 2008-{rep['years'][1]}, each predicted with his own class left out",
    color=MUTED,
    fontsize=8.5,
)
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(f"{OUT}/calibration.png")

# 2. late picks the model liked: what they did against what their slots said
prod = rep["skaters"]  # production only: no scouting input at all
liked = prod["late_picks_model_liked"]
fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=160)
labels = [
    "All picks after 60th",
    "Model's picks after 60th:\nwhat their draft slots predicted",
    "Model's picks after 60th:\nwhat they did",
]
vals = [prod["late_picks_all"]["played_share"], liked["draft_order_expected_share"], liked["played_share"]]
cols = [DRAFT, DRAFT, MODEL]
bars = ax.barh(labels[::-1], vals[::-1], color=cols[::-1], height=0.55)
for b, v in zip(bars, vals[::-1], strict=True):
    ax.text(v + 0.01, b.get_y() + b.get_height() / 2, f"{v:.0%}", va="center", color=INK, fontsize=11)
ax.set_xlim(0, max(vals) * 1.25)
ax.xaxis.set_visible(False)
for s in ("top", "right", "bottom"):
    ax.spines[s].set_visible(False)
ax.set_title("Share who played an NHL game", color=INK, loc="left", fontsize=12)
fig.text(
    0.01,
    0.01,
    f"Skaters drafted 2005-{rep['years'][1]}; the model's top 30 of each class by expected games, "
    "production only",
    color=MUTED,
    fontsize=8.5,
)
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(f"{OUT}/late_picks.png")
print("charts written")
