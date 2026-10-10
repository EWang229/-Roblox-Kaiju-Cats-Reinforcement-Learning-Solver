"""Plot the expert-iteration training curve from results/train_log.txt -> docs/training_curve.png"""
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
its, mean, best, expert, val_it, val = [], [], [], [], [], []
for line in (root / "results" / "train_log.txt").read_text().splitlines():
    m = re.match(r"it\s+(\d+)\s+apprentice mean\s+(\d+)\s+best-of-\d+\s+(\d+)\s+after expert\s+(\d+)", line)
    if m:
        its.append(int(m[1])); mean.append(int(m[2])); best.append(int(m[3])); expert.append(int(m[4]))
    m = re.match(r"\s+validation best-of-256:\s+(.*)", line)
    if m and its:
        nums = [int(x.replace(",", "")) for x in m[1].split()]
        val_it.append(its[-1]); val.append(sum(nums) / len(nums))
fig, ax = plt.subplots(figsize=(8, 5.2))
ax.plot(its, expert, label="after the expert improves the best game (target)", color="#c0392b")
ax.plot(its, best, label="policy: best of 64 games (fresh boards)", color="#2c7fb8")
ax.plot(its, mean, label="policy: average game", color="#7fb3d5")
ax.plot(val_it, val, "o--", label="held-out boards: best of 256 (mean of 8)", color="#27ae60", ms=4)
ax.set_xlabel("training iteration (16 new random boards each)")
ax.set_ylabel("score")
ax.set_title("Expert iteration: the policy chases the expert")
ax.grid(alpha=.3)
ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False)
fig.tight_layout()
fig.savefig(root / "docs" / "training_curve.png", dpi=140)
print("written", len(its), "points")
