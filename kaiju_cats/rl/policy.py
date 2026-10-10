"""
The policy network.

Input  : the board as an 18-channel 5x7 image plus a 41-number vector (see ``batch._observe``).
Output : 7 logits, one per command, for the cat that just broke a floor.

Design choices (and why)
------------------------
* **Convolutions, not a flat MLP.**  Whether a command is good depends on *geometry*: what is in front of
  the cat, whether a plant is two tiles away, where the bed is relative to the cat.  A convolution applies
  the same small "what is next to me" detector at every tile, so a pattern learned on one board transfers to
  another.  A flat MLP would have to relearn it for every position.
* **Padding 1, 3x3 kernels, 4 layers.**  Each layer widens what a tile "sees" by one step; four layers see most
  of the 5x7 board, which is all the context a decision needs.
* **Residual connections + LayerNorm-free design.**  Residual blocks keep gradients healthy for a net this
  small, and the board is tiny, so no normalisation layer is needed (fewer moving parts to go wrong).
* **Read the features at the cat's own tile.**  The decision is about *this* cat on *this* tile, so the head
  reads the feature map where channel 10 ("self") is 1, and adds a global average so far-away information
  (the bed on the other side) is available too.
* **Masked softmax.**  Commands the environment says are pointless or unaffordable get logit -inf, so the
  network never wastes probability on them and never needs to learn the budget rule.
* **Policy only, no value head.**  The training algorithm (group baseline, see train.py) does not need a critic.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .batch import N_ACTIONS, N_CHANNELS, N_VECTOR


class ResBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.c1 = nn.Conv2d(width, width, 3, padding=1)
        self.c2 = nn.Conv2d(width, width, 3, padding=1)

    def forward(self, x):
        return F.gelu(x + self.c2(F.gelu(self.c1(x))))


class PolicyNet(nn.Module):
    def __init__(self, width: int = 48, blocks: int = 1, hidden: int = 128):
        super().__init__()
        self.stem = nn.Conv2d(N_CHANNELS, width, 3, padding=1)
        self.blocks = nn.Sequential(*[ResBlock(width) for _ in range(blocks)])
        self.vec = nn.Sequential(nn.Linear(N_VECTOR, hidden), nn.GELU())
        self.head = nn.Sequential(nn.Linear(2 * width + hidden, hidden), nn.GELU(), nn.Linear(hidden, N_ACTIONS))
        self.width = width

    def forward(self, img: torch.Tensor, vec: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Return masked log-probabilities, shape (batch, 7)."""
        x = self.blocks(F.gelu(self.stem(img)))
        self_pos = img[:, 10:11]                                   # 1 where the deciding cat stands
        at_cat = (x * self_pos).sum(dim=(2, 3))                    # feature vector at the cat's tile
        pooled = x.mean(dim=(2, 3))                                # whole-board summary
        h = torch.cat([at_cat, pooled, self.vec(vec)], dim=1)
        logits = self.head(h).masked_fill(~mask, float("-inf"))
        return F.log_softmax(logits, dim=1)
