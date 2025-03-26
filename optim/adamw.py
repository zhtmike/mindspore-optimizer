from typing import List, Tuple

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.mint as mint
import numpy as np
from mindspore import Parameter, ParameterTuple, Tensor

_adam_opt = ops.MultitypeFuncGraph("adam_opt")


@_adam_opt.register(
    "Number",
    "Number",
    "Tensor",
    "Tensor",
    "Number",
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
    beta1: float,
    beta2: float,
    beta1_t: Parameter,
    beta2_t: Parameter,
    eps: float,
    lr: Tensor,
    weight_decay: Tensor,
    param: Parameter,
    m: Parameter,
    v: Parameter,
    gradient: Tensor,
    decay_flag: bool,
    optim_filter: bool,
) -> bool:
    if not optim_filter:
        return False

    if decay_flag:
        param.add_(-lr * weight_decay * param)

    m_next = mint.lerp(gradient, m, beta1)
    v_next = mint.lerp(mint.square(gradient), v, beta2)

    m_hat = m_next / (1 - beta1_t)
    v_hat = v_next / (1 - beta2_t)

    u = m_hat / (mint.sqrt(v_hat) + eps)
    param.add_(-lr * u)

    ops.assign(m, m_next)
    ops.assign(v, v_next)
    return True


class AdamW(nn.Optimizer):
    """Following https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html"""

    def __init__(
        self,
        params: List[Parameter],
        lr: float = 0.001,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        super().__init__(lr, params, weight_decay)
        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.eps = eps
        self.moments1 = ParameterTuple(
            [
                Parameter(np.zeros(x.shape, dtype=np.float32), name="m." + x.name)
                for x in self._parameters
            ]
        )
        self.moments2 = ParameterTuple(
            [
                Parameter(np.zeros(x.shape, dtype=np.float32), name="v." + x.name)
                for x in self._parameters
            ]
        )

        self.beta1_t = Parameter(Tensor(1, dtype=ms.float32))
        self.beta2_t = Parameter(Tensor(1, dtype=ms.float32))

    @ms.jit
    def construct(self, gradients: List[Tensor]) -> bool:
        weight_decay = self.get_weight_decay()
        lr = self.get_lr()
        self.assignadd(self.global_step, self.global_step_increase_tensor)

        self.beta1_t = self.beta1_t * self.beta1
        self.beta2_t = self.beta2_t * self.beta2

        if self.is_group:
            if self.is_group_lr:
                optim_result = self.hyper_map(
                    ops.partial(
                        _adam_opt,
                        self.beta1,
                        self.beta2,
                        self.beta1_t,
                        self.beta2_t,
                        self.eps,
                    ),
                    lr,
                    weight_decay,
                    self._parameters,
                    self.moments1,
                    self.moments2,
                    gradients,
                    self.decay_flags,
                    self.optim_filter,
                )
            else:
                optim_result = self.hyper_map(
                    ops.partial(
                        _adam_opt,
                        self.beta1,
                        self.beta2,
                        self.beta1_t,
                        self.beta2_t,
                        self.eps,
                        lr,
                    ),
                    weight_decay,
                    self._parameters,
                    self.moments1,
                    self.moments2,
                    gradients,
                    self.decay_flags,
                    self.optim_filter,
                )
        else:
            optim_result = self.hyper_map(
                ops.partial(
                    _adam_opt,
                    self.beta1,
                    self.beta2,
                    self.beta1_t,
                    self.beta2_t,
                    self.eps,
                    lr,
                    weight_decay,
                ),
                self._parameters,
                self.moments1,
                self.moments2,
                gradients,
                self.decay_flags,
                self.optim_filter,
            )

        return optim_result
