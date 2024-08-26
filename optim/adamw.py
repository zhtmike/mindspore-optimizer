from typing import List, Tuple

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Parameter, Tensor, ParameterTuple

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
    beta1_t: Parameter,
    beta2_t: Parameter,
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
    gradient = ops.cast(gradient, ms.float32)

    if decay_flag:
        param_fp32 = param_fp32 - lr * weight_decay * param_fp32

    ops.assign(m, beta1 * m + (1 - beta1) * gradient)
    ops.assign(v, beta2 * v + (1 - beta2) * ops.square(gradient))

    m_hat = m / (1 - beta1_t)
    v_hat = v / (1 - beta2_t)

    param_fp32 = param_fp32 - lr * m_hat / (ops.sqrt(v_hat) + eps)
    ops.assign(param, param_fp32.to(param.dtype))
    return param


class AdamW(nn.Optimizer):
    """Following https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html"""

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
        self.moments1 = ParameterTuple(
            [
                Parameter(ops.zeros_like(x, dtype=ms.float32), name=x.name + "_m")
                for x in self._parameters
            ]
        )
        self.moments2 = ParameterTuple(
            [
                Parameter(ops.zeros_like(x, dtype=ms.float32), name=x.name + "_v")
                for x in self._parameters
            ]
        )

        self.beta1_t = Parameter(Tensor(1, dtype=ms.float32))
        self.beta2_t = Parameter(Tensor(1, dtype=ms.float32))

    @ms.jit
    def construct(self, gradients: List[Tensor]):
        gradients = self.flatten_gradients(gradients)
        weight_decay = self.get_weight_decay()
        lr = self.get_lr()
        self.assignadd(self.global_step, self.global_step_increase_tensor)

        ops.assign(self.beta1_t, self.beta1_t * self.beta1)
        ops.assign(self.beta2_t, self.beta2_t * self.beta2)

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

        if self.use_parallel:
            self.broadcast_params(optim_result)

        return optim_result
