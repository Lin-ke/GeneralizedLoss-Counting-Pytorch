import sys, os.path

__version__ = '0.2.3'

from .samples_loss import SamplesLoss
from .samples_loss import ot_sinkhorn, uot_sinkhorn

__all__ = sorted(["SamplesLoss"])
