# -*- coding: utf-8 -*-
"""Plot the grokking accuracy curve (paper Figure 1, right panel only).

Usage:
    python src/plot_acc.py                      # auto-discover latest run
    python src/plot_acc.py --csv PATH           # plot a specific train_log.csv
"""
import argparse
import csv
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "..", "runs")


def latest_run():
    """Find the newest run directory that has a train_log.csv."""
    logs = glob.glob(os.path.join(RUNS, "*", "train_log.csv"))
    if not logs:
        raise SystemExit("no train_log.csv found under runs/ -- train first")
    return os.path.dirname(max(logs, key=os.path.getmtime))


def load(log):
    steps, tr, va = [], [], []
    with open(log, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            steps.append(int(row[0]))
            tr.append(float(row[2]))
            va.append(float(row[4]))
    return steps, tr, va


ap = argparse.ArgumentParser()
ap.add_argument("--csv", type=str, default=None,
                help="path to a specific train_log.csv (default: auto-discover)")
ap.add_argument("--out", type=str, default=None,
                help="output png path (default: runs/grokking_acc.png)")
ap.add_argument("--title", type=str, default=None,
                help="run name shown in the title")
args = ap.parse_args()

run_dir = os.path.dirname(os.path.abspath(args.csv)) if args.csv else latest_run()
s, tr, va = load(os.path.join(run_dir, "train_log.csv"))
name = args.title or os.path.basename(run_dir)
print(f"plotting run: {name} ({len(s)} log points)")

C_TR, C_VA = "#d62728", "#2ca02c"   # paper color scheme: train=red, val=green

fig, ax = plt.subplots(figsize=(7, 4.8))

ax.plot(s, tr, color=C_TR, lw=2, label="train acc")
ax.plot(s, va, color=C_VA, lw=2, label="val acc")
ax.axhline(1.0, color="gray", ls="--", alpha=0.5)
ax.axhline(1 / 97, color="gray", ls=":", alpha=0.6)
ax.text(1.3, 1 / 97 + 0.03, "random chance (1/97)", fontsize=8, color="gray")
ax.set_xscale("log")
ax.set_xlabel("optimization steps", fontsize=11)
ax.set_ylabel("accuracy", fontsize=11)
ax.set_ylim(-0.03, 1.08)
ax.set_title(name, fontsize=12)
ax.legend(loc="lower right", fontsize=10)
ax.grid(alpha=0.3, which="both")

# annotate key milestones if present
tr_full = next((x for x, y in zip(s, tr) if y >= 0.999), None)
va_full = next((x for x, y in zip(s, va) if y >= 0.999), None)
if tr_full:
    ax.annotate(f"train 100% (step {tr_full})", xy=(tr_full, 1.0),
                xytext=(max(s) * 0.001, 0.60), fontsize=9,
                arrowprops=dict(arrowstyle="->", color="gray", alpha=0.7))
if va_full:
    ax.annotate(f"grokking!\nval 100% (step {va_full})", xy=(va_full, 1.0),
                xytext=(max(s) * 0.02, 0.30), fontsize=9,
                arrowprops=dict(arrowstyle="->", color="black", alpha=0.8))

fig.tight_layout()
out = args.out or os.path.join(RUNS, "grokking_acc.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved:", out)
