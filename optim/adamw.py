from typing import Tuple, List
import mindspore as ms
import mindspore.nn as nn
from mindspore import Parameter


class AdamW(nn.Optimizer):
    def __init__(
        self,
        params: List[],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
    ): ...
