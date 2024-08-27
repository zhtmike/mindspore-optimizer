from typing import List, Optional, Tuple

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Parameter, ParameterTuple, Tensor

_adafactor_opt = ops.MultitypeFuncGraph("adafactor_opt")


@_adafactor_opt.register(
    "Tensor",
    "Tensor",
    "Tensor",
    "Tensor",
    "Tensor",
    "Bool",
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
    eps1: Tensor,
    eps2: Tensor,
    d: Tensor,
    use_first_moment: bool,
    rho: Tensor,
    weight_decay: Tensor,
    param: Parameter,
    m: Parameter,
    v_row: Parameter,
    v_col: Parameter,
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

    alpha = ops.maximum(eps2, _rms(param_)) * rho
    update = ops.square(gradient) + eps1

    v_row_next, v_col_next, v_next = None, None, None
    factored = len(gradient.shape) >= 2
    if factored:
        v_row_next = beta2 * v_row + (1 - beta2) * ops.mean(update, axis=-1)
        v_col_next = beta2 * v_col + (1 - beta2) * ops.mean(update, axis=-2)
        u = _approx_sq_grad(v_row_next, v_col_next)
        u = u * gradient
    else:
        v_next = beta2 * v + (1 - beta2) * update
        u = ops.rsqrt(v_next) * gradient

    u = u / ops.clamp(_rms(u) / d, min=1.0)

    m_next = None
    if use_first_moment:
        m_next = beta1 * m + (1 - beta1) * u
        u = m

    param_ = param_ - alpha * u

    if decay_flag:
        param_ = param_ - alpha * weight_decay * param_

    param_ = ops.cast(param_, dtype)
    ops.assign(param, param_)
    if factored:
        ops.assign(v_row, v_row_next)
        ops.assign(v_col, v_col_next)
    else:
        ops.assign(v, v_next)

    if use_first_moment:
        ops.assign(m, m_next)

    return param_


def _rms(x: Tensor) -> Tensor:
    return ops.sqrt(ops.mean(ops.square(x)))


def _approx_sq_grad(v_row: Tensor, v_col: Tensor) -> Tensor:
    r_factor = v_row / ops.mean(v_row, axis=-1, keep_dims=True)
    r_factor = ops.rsqrt(r_factor)
    r_factor = ops.unsqueeze(r_factor, -1)
    c_factor = ops.unsqueeze(v_col, -2)
    c_factor = ops.rsqrt(c_factor)
    return ops.mul(r_factor, c_factor)


class AdaFactor(nn.Optimizer):
    """Following https://huggingface.co/docs/transformers/main_classes/optimizer_schedules#transformers.Adafactor"""

    _support_parallel_optimizer = True

    def __init__(
        self,
        params: List[Parameter],
        lr: Optional[float] = None,
        eps: Tuple[float, float] = (1e-30, 1e-3),
        clip_threshold: float = 1.0,
        decay_rate: float = -0.8,
        beta1: Optional[float] = None,
        weight_decay: float = 0.0,
        relative_step: bool = True,
    ) -> None:
        super().__init__(lr, params, weight_decay)

        if not relative_step and lr is None:
            raise ValueError("`lr should be provided when `relatvie_step` is `False`.")
        elif relative_step and lr is not None:
            raise ValueError("`lr should be None when `relatvie_step` is `True`.")

        self.eps1 = Tensor(eps[0], dtype=ms.float32)
        self.eps2 = Tensor(eps[1], dtype=ms.float32)
        self.clip_threshold = Tensor(clip_threshold, dtype=ms.float32)
        self.decay_rate = Tensor(decay_rate, dtype=ms.float32)
        self.relatvie_step = relative_step
        if beta1 is None:
            self.beta1 = Tensor(0.0, dtype=ms.float32)
            self.use_first_moment = False
        else:
            self.beta1 = Tensor(beta1, dtype=ms.float32)
            self.use_first_moment = True

        v_row, v_col, v = list(), list(), list()
        for x in self._parameters:
            if len(x.shape) >= 2:
                v_row.append(
                    Parameter(
                        ops.zeros(x.shape[:-1], dtype=ms.float32),
                        name=x.name + "_v_row",
                        requires_grad=False,
                    )
                )
                v_col.append(
                    Parameter(
                        ops.zeros(x.shape[:-2] + x.shape[-1:], dtype=ms.float32),
                        name=x.name + "_v_col",
                        requires_grad=False,
                    )
                )
                v.append(
                    Parameter(
                        ops.zeros((1,), dtype=ms.float32),
                        name=x.name + "_v",
                        requires_grad=False,
                    )
                )
            else:
                v_row.append(
                    Parameter(
                        ops.zeros((1,), dtype=ms.float32),
                        name=x.name + "_v_row",
                        requires_grad=False,
                    )
                )
                v_col.append(
                    Parameter(
                        ops.zeros((1,), dtype=ms.float32),
                        name=x.name + "_v_col",
                        requires_grad=False,
                    )
                )
                v.append(
                    Parameter(
                        ops.zeros_like(x, dtype=ms.float32),
                        name=x.name + "_v",
                        requires_grad=False,
                    )
                )

        self.v_row = ParameterTuple(v_row)
        self.v_col = ParameterTuple(v_col)
        self.v = ParameterTuple(v)

        if self.use_first_moment:
            self.m = ParameterTuple(
                [
                    Parameter(
                        ops.zeros_like(x, dtype=ms.float32),
                        name=x.name + "_m",
                        requires_grad=False,
                    )
                    for x in self._parameters
                ]
            )
        else:
            self.m = ParameterTuple(
                [
                    Parameter(
                        ops.zeros((1,), dtype=ms.float32),
                        name=x.name + "_m",
                        requires_grad=False,
                    )
                    for x in self._parameters
                ]
            )

    def _preprocess_single_lr(self, learning_rate):
        try:
            return super()._preprocess_single_lr(learning_rate)
        except TypeError:
            if learning_rate is None:
                return None
            raise

    def get_lr(self):
        if self.learning_rate is None:
            return None
        return super().get_lr()

    @ms.jit
    def construct(self, gradients: List[Tensor]):
        gradients = self.flatten_gradients(gradients)
        weight_decay = self.get_weight_decay()
        lr = self.get_lr()
        self.assignadd(self.global_step, self.global_step_increase_tensor)

        if self.relatvie_step:
            rho = ops.minimum(1e-2, 1.0 / ops.sqrt(self.global_step))
        else:
            rho = lr

        beta2 = 1.0 - ops.pow(self.global_step, self.decay_rate)

        if self.is_group:
            if self.is_group_lr:
                optim_result = self.hyper_map(
                    ops.partial(
                        _adafactor_opt,
                        self.beta1,
                        beta2,
                        self.eps1,
                        self.eps2,
                        self.clip_threshold,
                        self.use_first_moment,
                    ),
                    rho,
                    weight_decay,
                    self._parameters,
                    self.m,
                    self.v_row,
                    self.v_col,
                    self.v,
                    gradients,
                    self.decay_flags,
                    self.optim_filter,
                )
            else:
                optim_result = self.hyper_map(
                    ops.partial(
                        _adafactor_opt,
                        self.beta1,
                        beta2,
                        self.eps1,
                        self.eps2,
                        self.clip_threshold,
                        self.use_first_moment,
                        rho,
                    ),
                    weight_decay,
                    self._parameters,
                    self.m,
                    self.v_row,
                    self.v_col,
                    self.v,
                    gradients,
                    self.decay_flags,
                    self.optim_filter,
                )
        else:
            optim_result = self.hyper_map(
                ops.partial(
                    _adafactor_opt,
                    self.beta1,
                    beta2,
                    self.eps1,
                    self.eps2,
                    self.clip_threshold,
                    self.use_first_moment,
                    rho,
                    weight_decay,
                ),
                self._parameters,
                self.m,
                self.v_row,
                self.v_col,
                self.v,
                gradients,
                self.decay_flags,
                self.optim_filter,
            )

        if self.use_parallel:
            self.broadcast_params(optim_result)

        return optim_result
