from typing import List

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.mint as mint
import numpy as np
from mindspore import Parameter, ParameterTuple, Tensor

_rmsprop_opt = ops.MultitypeFuncGraph("rmsprop_opt")


@_rmsprop_opt.register(
    "Tensor",
    "Tensor",
    "Tensor",
    "Tensor",
    "Tensor",
    "Tensor",
    "Tensor",
    "Bool",
    "Bool",
)
def _update_run_op(
    alpha: Tensor,
    eps: Tensor,
    lr: Tensor,
    weight_decay: Tensor,
    param: Parameter,
    v: Parameter,
    gradient: Tensor,
    decay_flag: bool,
    optim_filter: bool,
) -> Tensor:
    if not optim_filter:
        return gradient

    dtype = param.dtype
    param_ = ops.cast(param, ms.float32)
    gradient = ops.cast(gradient, ms.float32)

    if decay_flag:
        gradient = mint.add(gradient, param_, alpha=weight_decay)

    v_next = mint.lerp(mint.square(gradient), v, alpha)
    u = gradient / (mint.sqrt(v_next) + eps)

    param_ = mint.add(param_, u, alpha=-lr)
    param_ = ops.cast(param_, dtype)
    ops.assign(param, param_)
    ops.assign(v, v_next)
    return param_


class RMSprop(nn.Optimizer):
    """Following https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html"""

    def __init__(
        self,
        params: List[Parameter],
        lr: float = 0.001,
        alpha: float = 0.99,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(lr, params, weight_decay)
        self.alpha = Tensor(alpha, dtype=ms.float32)
        self.eps = Tensor(eps, dtype=ms.float32)
        self.moments2 = ParameterTuple(
            [
                Parameter(np.zeros(x.shape, dtype=np.float32), name="v." + x.name)
                for x in self._parameters
            ]
        )

    @ms.jit
    def construct(self, gradients: List[Tensor]):
        weight_decay = self.get_weight_decay()
        lr = self.get_lr()
        self.assignadd(self.global_step, self.global_step_increase_tensor)

        if self.is_group:
            if self.is_group_lr:
                optim_result = self.hyper_map(
                    ops.partial(
                        _rmsprop_opt,
                        self.alpha,
                        self.eps,
                    ),
                    lr,
                    weight_decay,
                    self._parameters,
                    self.moments2,
                    gradients,
                    self.decay_flags,
                    self.optim_filter,
                )
            else:
                optim_result = self.hyper_map(
                    ops.partial(
                        _rmsprop_opt,
                        self.alpha,
                        self.eps,
                        lr,
                    ),
                    weight_decay,
                    self._parameters,
                    self.moments2,
                    gradients,
                    self.decay_flags,
                    self.optim_filter,
                )
        else:
            optim_result = self.hyper_map(
                ops.partial(
                    _rmsprop_opt,
                    self.alpha,
                    self.eps,
                    lr,
                    weight_decay,
                ),
                self._parameters,
                self.moments2,
                gradients,
                self.decay_flags,
                self.optim_filter,
            )

        return optim_result
