from typing import List

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
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
        gradient = gradient + weight_decay * param_

    v_next = alpha * v + (1 - alpha) * ops.square(gradient)

    param_ = param_ - lr * gradient / (ops.sqrt(v_next) + eps)
    param_ = ops.cast(param_, dtype)
    ops.assign(param, param_)
    ops.assign(v, v_next)
    return param_


class RMSprop(nn.Optimizer):
    """Following https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html"""

    _support_parallel_optimizer = True

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
                Parameter(
                    ops.zeros_like(x, dtype=ms.float32),
                    name=x.name + "_v",
                    requires_grad=False,
                )
                for x in self._parameters
            ]
        )

    @ms.jit
    def construct(self, gradients: List[Tensor]):
        gradients = self.flatten_gradients(gradients)
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

        if self.use_parallel:
            self.broadcast_params(optim_result)

        return optim_result
