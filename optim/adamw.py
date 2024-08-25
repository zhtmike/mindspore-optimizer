from typing import List, Tuple

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Parameter, Tensor

_adam_opt = ops.MultitypeFuncGraph("adam_opt")


@_adam_opt.register(
    "Tensor",
    "Tensor",
    "Tensor",
    "Tensor",
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
    beta1: Tensor,
    beta2: Tensor,
    beta1_power: Parameter,
    beta2_power: Parameter,
    eps: Tensor,
    lr: Tensor,
    weight_decay: Tensor,
    param: Parameter,
    m: Parameter,
    v: Parameter,
    gradient: Tensor,
    decay_flag: bool,
    optim_filter: bool,
):
    if not optim_filter:
        return gradient

    param_fp32 = ops.cast(param, ms.float32)
    m_fp32 = ops.cast(m, ms.float32)
    v_fp32 = ops.cast(v, ms.float32)
    gradient = gradient.to(ms.float32)

    if decay_flag:
        param_fp32 = param_fp32 - lr * weight_decay * param_fp32

    m_fp32 = beta1 * m_fp32 + (1 - beta1) * gradient
    v_fp32 = beta2 * v_fp32 + (1 - beta2) * ops.square(gradient)
    ops.assign(m, m_fp32.to(m.dtype))
    ops.assign(v, v_fp32.to(v.dtype))

    m_fp32 = m_fp32 / (1 - beta1_power)
    v_fp32 = v_fp32 / (1 - beta2_power)

    param_fp32 = param_fp32 - lr * m_fp32 / (ops.sqrt(v_fp32) + eps)
    ops.assign(param, param_fp32.to(param.dtype))
    return param


class AdamW(nn.Optimizer):
    _support_parallel_optimizer = True

    def __init__(
        self,
        params: List[Parameter],
        lr: float = 0.001,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        super().__init__(lr, params, weight_decay)
        self.beta1 = Tensor(betas[0], dtype=ms.float32)
        self.beta2 = Tensor(betas[1], dtype=ms.float32)
        self.eps = Tensor(eps, dtype=ms.float32)
        self.moments1 = self._parameters.clone(prefix="adam_m", init="zeros")
        self.moments2 = self._parameters.clone(prefix="adam_v", init="zeros")

        self.beta1_power = Parameter(Tensor(1, dtype=ms.float32))
        self.beta2_power = Parameter(Tensor(1, dtype=ms.float32))

    @ms.jit
    def construct(self, gradients: List[Tensor]):
        gradients = self.flatten_gradients(gradients)
        weight_decay = self.get_weight_decay()
        lr = self.get_lr()
        self.assignadd(self.global_step, self.global_step_increase_tensor)

        ops.assign(self.beta1_power, self.beta1_power * self.beta1)
        ops.assign(self.beta2_power, self.beta2_power * self.beta1)

        if self.is_group:
            if self.is_group_lr:
                optim_result = self.hyper_map(
                    ops.partial(
                        _adam_opt,
                        self.beta1,
                        self.beta2,
                        self.beta1_power,
                        self.beta2_power,
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
                        self.beta1_power,
                        self.beta2_power,
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
                    self.beta1_power,
                    self.beta2_power,
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

        if self.use_parallel:
            self.broadcast_params(optim_result)

        return optim_result
