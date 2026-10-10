"""Train the policy with expert iteration on freshly generated random boards.

Example:  python scripts/train_exit.py --iterations 600 --out runs/exit.pt
"""
import argparse
import glob

from kaiju_cats import Board
from kaiju_cats.rl.exit import ExItConfig, train_exit

ap = argparse.ArgumentParser()
ap.add_argument("--iterations", type=int, default=600)
ap.add_argument("--boards", type=int, default=16)
ap.add_argument("--rollouts", type=int, default=64)
ap.add_argument("--top", type=int, default=3)
ap.add_argument("--depth", type=int, default=2)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--updates", type=int, default=8)
ap.add_argument("--pg", type=float, default=0.0)
ap.add_argument("--width", type=int, default=48)
ap.add_argument("--exact", type=int, default=100000)
ap.add_argument("--blocks", type=int, default=1)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--val-every", type=int, default=50)
ap.add_argument("--out", default="runs/exit.pt")
a = ap.parse_args()

val = [Board.from_file(f) for f in sorted(glob.glob("boards_val/*.txt"))]
cfg = ExItConfig(iterations=a.iterations, boards_per_iter=a.boards, rollouts=a.rollouts, top=a.top,
                 expert_depth=a.depth, exact_nodes=a.exact, lr=a.lr, updates=a.updates, pg_weight=a.pg, seed=a.seed, save_path=a.out, width=a.width,
                 blocks=a.blocks, val_every=a.val_every)
train_exit(cfg, val_boards=val, log=lambda s: print(s, flush=True))
