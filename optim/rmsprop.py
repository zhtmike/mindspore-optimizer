from typing import List

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.mint as mint
import numpy as np
from mindspore import Parameter, ParameterTuple, Tensor

_rmsprop_opt = ops.MultitypeFuncGraph("rmsprop_opt")


@_rmsprop_opt.register(
    "Number",
    "Number",
    "Number",
    "Bool",
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
    alpha: float,
    eps: float,
    momentum: float,
    centered: bool,
    lr: Tensor,
    weight_decay: Tensor,
    param: Parameter,
    v: Parameter,
    b: Parameter,
    g_ave: Parameter,
    g: Tensor,
    decay_flag: bool,
    optim_filter: bool,
) -> bool:
    if not optim_filter:
        return False

    if decay_flag:
        g.add_(weight_decay * param)

    v_next = mint.lerp(mint.square(g), v, alpha)

    if centered:
        g_ave_next = mint.lerp(g, g_ave)
        v_next.add_(-mint.square(g_ave_next)) 

    if momentum > 0:
        g = momentum * b + g / (mint.sqrt(v_next) + eps)
    else:
        g = g / (mint.sqrt(v_next) + eps)

    param.add_(- lr * g)

    ops.assign(v, v_next)
    if momentum > 0:
        ops.assign(b, g)
    if centered:
        ops.assign(g_ave, g_ave_next)
    return True


class RMSprop(nn.Optimizer):
    """Following https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html"""

    def __init__(
        self,
        params: List[Parameter],
        lr: float = 0.001,
        alpha: float = 0.99,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        momentum: float = 0.0,
        centered: bool = False,
    ) -> None:
        super().__init__(lr, params, weight_decay)
        self.alpha = alpha
        self.eps = eps
        self.momentum = momentum
        self.centered = centered
        self.moments2 = ParameterTuple(
            [
                Parameter(np.zeros(x.shape, dtype=np.float32), name="v." + x.name)
                for x in self._parameters
            ]
        )
        if self.momentum > 0:
            self.moments1 = ParameterTuple(
                [
                    Parameter(np.zeros(x.shape, dtype=np.float32), name="b." + x.name)
                    for x in self._parameters
                ]
            )
        else:
            self.moments1 = ParameterTuple(
                [
                    Parameter([], name="b." + x.name)
                    for x in self._parameters
                ]
            )
        
        if self.centered:
            self.gradient_ave = ParameterTuple(
                [
                    Parameter(np.zeros(x.shape, dtype=np.float32), name="g_ave." + x.name)
                    for x in self._parameters
                ]
            )
        else:
            self.gradient_ave = ParameterTuple(
                [
                    Parameter([], name="g_ave." + x.name)
                    for x in self._parameters
                ]
            )

    @ms.jit
    def construct(self, gradients: List[Tensor]) -> bool:
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
                        self.momentum,
                        self.centered,
                    ),
                    lr,
                    weight_decay,
                    self._parameters,
                    self.moments2,
                    self.moments1,
                    self.gradient_ave,
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
                        self.momentum,
                        self.centered,
                        lr,
                    ),
                    weight_decay,
                    self._parameters,
                    self.moments2,
                    self.moments1,
                    self.gradient_ave,
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
                self.moments1,
                self.gradient_ave,
                gradients,
                self.decay_flags,
                self.optim_filter,
            )

        return optim_result
