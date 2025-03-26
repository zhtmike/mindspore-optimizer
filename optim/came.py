from typing import List, Tuple

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.mint as mint
import numpy as np
from mindspore import Parameter, ParameterTuple, Tensor

_came_opt = ops.MultitypeFuncGraph("came_opt")


@_came_opt.register(
    "Number",
    "Number",
    "Number",
    "Number",
    "Number",
    "Number",
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
    beta1: float,
    beta2: float,
    beta3: float,
    eps1: float,
    eps2: float,
    d: float,
    lr: Tensor,
    weight_decay: Tensor,
    param: Parameter,
    m: Parameter,
    v_row: Parameter,
    v_col: Parameter,
    v_res_row: Parameter,
    v_res_col: Parameter,
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
        param_ = param_ - lr * weight_decay * param_

    update = mint.square(gradient) + eps1

    v_row_next, v_col_next, v_next = None, None, None
    factored = len(gradient.shape) >= 2
    if factored:
        v_row_next = mint.lerp(mint.mean(update, dim=-1), v_row, beta2)
        v_col_next = mint.lerp(mint.mean(update, dim=-2), v_col, beta2)
        u = _approx_sq_grad(v_row_next, v_col_next)
        u = u * gradient
    else:
        v_next = mint.lerp(update, v, beta2)
        u = mint.rsqrt(v_next) * gradient

    u = u / mint.clamp(_rms(u) / d, min=1.0)

    m_next = mint.lerp(u, m, beta1)

    v_res_row_next, v_res_col_next = None, None
    if factored:
        res = mint.square(u - m_next) + eps2
        v_res_row_next = mint.lerp(mint.mean(res, dim=-1), v_res_row, beta3)
        v_res_col_next = mint.lerp(mint.mean(res, dim=-2), v_res_col, beta3)
        u = _approx_sq_grad(v_res_row_next, v_res_col_next)
        u = u * m_next
    else:
        u = m_next

    param_ = param_ - lr * u

    param_ = ops.cast(param_, dtype)
    ops.assign(param, param_)
    ops.assign(m, m_next)
    if factored:
        ops.assign(v_row, v_row_next)
        ops.assign(v_col, v_col_next)
        ops.assign(v_res_row, v_res_row_next)
        ops.assign(v_res_col, v_res_col_next)
    else:
        ops.assign(v, v_next)

    return param_


def _rms(x: Tensor) -> Tensor:
    return mint.norm(x, p=2) / (x.numel() ** 0.5)


def _approx_sq_grad(v_row: Tensor, v_col: Tensor) -> Tensor:
    r_factor = v_row / mint.mean(v_row, dim=-1, keepdim=True)
    r_factor = mint.rsqrt(r_factor)
    r_factor = mint.unsqueeze(r_factor, -1)
    c_factor = mint.unsqueeze(v_col, -2)
    c_factor = mint.rsqrt(c_factor)
    return mint.mul(r_factor, c_factor)


class CAME(nn.Optimizer):
    """Following https://github.com/yangluo7/CAME"""

    def __init__(
        self,
        params: List[Parameter],
        lr: float = 2e-4,
        eps: Tuple[float, float] = (1e-30, 1e-16),
        clip_threshold: float = 1.0,
        betas: Tuple[float, float, float] = (0.9, 0.999, 0.9999),
        weight_decay: float = 0.01,
    ) -> None:
        super().__init__(lr, params, weight_decay)

        self.eps1 = eps[0]
        self.eps2 = eps[1]
        self.clip_threshold = clip_threshold
        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.beta3 = betas[2]

        v_row, v_col, v_res_row, v_res_col, v = list(), list(), list(), list(), list()
        for x in self._parameters:
            if len(x.shape) >= 2:
                v_row.append(
                    Parameter(
                        np.zeros(x.shape[:-1], dtype=np.float32), name="v_row." + x.name
                    )
                )
                v_col.append(
                    Parameter(
                        np.zeros(x.shape[:-2] + x.shape[-1:], dtype=np.float32),
                        name="v_col." + x.name,
                    )
                )
                v_res_row.append(
                    Parameter(
                        np.zeros(x.shape[:-1], dtype=np.float32),
                        name="v_res_row." + x.name,
                    )
                )
                v_res_col.append(
                    Parameter(
                        np.zeros(x.shape[:-2] + x.shape[-1:], dtype=np.float32),
                        name="v_res_col." + x.name,
                    )
                )
                v.append(Parameter([], name="v." + x.name))
            else:
                v_row.append(Parameter([], name="v_row." + x.name))
                v_col.append(Parameter([], name="v_col." + x.name))
                v_res_row.append(Parameter([], name="v_res_row." + x.name))
                v_res_col.append(Parameter([], name="v_res_col." + x.name))
                v.append(
                    Parameter(np.zeros(x.shape, dtype=np.float32), name="v." + x.name)
                )

        self.v_row = ParameterTuple(v_row)
        self.v_col = ParameterTuple(v_col)
        self.v_res_row = ParameterTuple(v_res_row)
        self.v_res_col = ParameterTuple(v_res_col)
        self.v = ParameterTuple(v)

        self.m = ParameterTuple(
            [
                Parameter(np.zeros(x.shape, dtype=np.float32), name="m." + x.name)
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
                        _came_opt,
                        self.beta1,
                        self.beta2,
                        self.beta3,
                        self.eps1,
                        self.eps2,
                        self.clip_threshold,
                    ),
                    lr,
                    weight_decay,
                    self._parameters,
                    self.m,
                    self.v_row,
                    self.v_col,
                    self.v_res_row,
                    self.v_res_col,
                    self.v,
                    gradients,
                    self.decay_flags,
                    self.optim_filter,
                )
            else:
                optim_result = self.hyper_map(
                    ops.partial(
                        _came_opt,
                        self.beta1,
                        self.beta2,
                        self.beta3,
                        self.eps1,
                        self.eps2,
                        self.clip_threshold,
                        lr,
                    ),
                    weight_decay,
                    self._parameters,
                    self.m,
                    self.v_row,
                    self.v_col,
                    self.v_res_row,
                    self.v_res_col,
                    self.v,
                    gradients,
                    self.decay_flags,
                    self.optim_filter,
                )
        else:
            optim_result = self.hyper_map(
                ops.partial(
                    _came_opt,
                    self.beta1,
                    self.beta2,
                    self.beta3,
                    self.eps1,
                    self.eps2,
                    self.clip_threshold,
                    lr,
                    weight_decay,
                ),
                self._parameters,
                self.m,
                self.v_row,
                self.v_col,
                self.v_res_row,
                self.v_res_col,
                self.v,
                gradients,
                self.decay_flags,
                self.optim_filter,
            )

        return optim_result
