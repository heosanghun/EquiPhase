import torch
import torch.nn as nn
from equiphase.models.damped_momentum_deq import DampedMomentumDEQ

# SymplecticDEQ is renamed to DampedMomentumDEQ pursuant to FREEZE_PAPER2.md condition 4
SymplecticDEQ = DampedMomentumDEQ
