"""Train the policy on freshly generated random boards.

Example:  python scripts/train_policy.py --iterations 3000 --out runs/policy.pt
The 8 real boards are never used for training; the generated validation boards in boards_val/ are the
progress check, and the real boards are the final test.
"""
import argparse
import glob

from kaiju_cats import Board
from kaiju_cats.rl.train import TrainConfig, train

ap = argparse.ArgumentParser()
ap.add_argument("--iterations", type=int, default=2000)
ap.add_argument("--boards", type=int, default=16)
ap.add_argument("--rollouts", type=int, default=32)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--entropy", type=float, default=0.05)
ap.add_argument("--width", type=int, default=32)
ap.add_argument("--blocks", type=int, default=1)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="runs/policy.pt")
ap.add_argument("--val-every", type=int, default=100)
a = ap.parse_args()

val = [Board.from_file(f) for f in sorted(glob.glob("boards_val/*.txt"))]
cfg = TrainConfig(iterations=a.iterations, boards_per_iter=a.boards, rollouts=a.rollouts, lr=a.lr,
                  entropy_start=a.entropy, seed=a.seed, save_path=a.out, width=a.width, blocks=a.blocks)
train(cfg, val_boards=val, val_every=a.val_every, log=lambda s: print(s, flush=True))
