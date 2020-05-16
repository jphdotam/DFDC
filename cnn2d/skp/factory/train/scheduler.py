from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts,
    CosineAnnealingLR,
    ReduceLROnPlateau,
    OneCycleLR,
    CyclicLR
)

from .onecycle import CustomOneCycleLR