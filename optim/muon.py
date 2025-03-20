from typing import List, Tuple

import math
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.mint as mint
import numpy as np
from mindspore import Parameter, ParameterTuple, Tensor


_muon_opt = ops.MultitypeFuncGraph("muon_opt")


@_muon_opt.register(
    "Tensor",
    "Tensor",
    "Tensor",
    "Tensor",
    "Tensor",
    "Tensor",
    "Bool",
    "Number",
    "Tensor",
    "Tensor",
    "Tensor",
    "Tensor",
    "Tensor",
    "Tensor",
    "Number",
    "Bool",
    "Bool",
    "Bool",
)
def _update_run_op(
    mu: Tensor,
    beta1: Tensor,
    beta2: Tensor,
    beta1_t: Parameter,
    beta2_t: Parameter,
    eps: Tensor,
    nesterov: bool,
    steps: int,
    lr: Parameter,
    weight_decay: Tensor,
    param: Parameter,
    m: Parameter,
    v: Parameter,
    gradient: Tensor,
    ratio: float,
    use_muon: bool,
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

    v_next = None
    if use_muon:
        # Muon branch
        m_next = mu * m + gradient
        if nesterov:
            g = mu * m_next + gradient
        else:
            g = m_next
        u = zeropower_via_newtonschulz5(g, steps=steps)
        param_ = param_ - lr * ratio * u
    else:
        # AdamW branch
        m_next = beta1 * m + (1 - beta1) * gradient
        v_next = beta2 * v + (1 - beta2) * mint.square(gradient)
        m_hat = m_next / (1 - beta1_t)
        v_hat = v_next / (1 - beta2_t)
        param_ = param_ - lr * m_hat / (mint.sqrt(v_hat) + eps)
    param_ = ops.cast(param_, dtype)
    ops.assign(param, param_)
    ops.assign(m, m_next)
    if not use_muon:
        ops.assign(v, v_next)
    return param_


def zeropower_via_newtonschulz5(G: Tensor, steps: int) -> Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.shape[0] > G.shape[1]:
        X = X.T
    # Ensure spectral norm is at most 1
    X = X / (mint.norm(X) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.T
        B = (
            b * A + c * A @ A
        )  # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.shape[0] > G.shape[1]:
        X = X.T
    return X


class Muon(nn.Optimizer):
    """Following https://github.com/MoonshotAI/Moonlight"""

    def __init__(
        self,
        params: List[Parameter],
        lr: float = 0.001,
        momentum: float = 0.95,
        ns_steps: int = 5,
        adamw_betas: Tuple[float, float] = (0.9, 0.999),
        adamw_eps: float = 1e-8,
        nesterov: bool = True,
        weight_decay: float = 0.1,
        adamw_parameter_names: Tuple[str, ...] = ("embed_tokens.", "lm_head."),
    ) -> None:
        super().__init__(lr, params, weight_decay)

        if not isinstance(adamw_parameter_names, (tuple, list)):
            raise ValueError("`adamw_parameter_names` must be a tuple or list.")

        self.momentum = Tensor(momentum, dtype=ms.float32)
        self.adamw_beta1 = Tensor(adamw_betas[0], dtype=ms.float32)
        self.adamw_beta2 = Tensor(adamw_betas[1], dtype=ms.float32)
        self.adamw_eps = Tensor(adamw_eps, dtype=ms.float32)
        self.moments1 = ParameterTuple(
            [
                Parameter(np.zeros(x.shape, dtype=np.float32), name="m." + x.name)
                for x in self._parameters
            ]
        )
        self.use_muon = tuple(
            [
                (
                    True
                    if len(x.shape) == 2
                    and not any([p in x.name for p in adamw_parameter_names])
                    else False
                )
                for x in self._parameters
            ]
        )
        self.moments2 = ParameterTuple(
            [
                (
                    Parameter(np.zeros(x.shape, dtype=np.float32), name="v." + x.name)
                    if not use_muon
                    else Parameter([], name="v." + x.name)
                )
                for x, use_muon in zip(self._parameters, self.use_muon)
            ]
        )
        self.adamw_beta1_t = Parameter(Tensor(1, dtype=ms.float32))
        self.adamw_beta2_t = Parameter(Tensor(1, dtype=ms.float32))
        self.ns_steps = ns_steps
        self.nesterov = nesterov

        self.lr_ratio = tuple([self._cal_lr_ratio(x) for x in self._parameters])

    def _cal_lr_ratio(self, param: Parameter) -> float:
        A, B = param.shape
        # We adjust the learning rate and weight decay based on the size of the parameter matrix
        # as describted in the paper
        adjusted_ratio = 0.2 * math.sqrt(max(A, B))
        return adjusted_ratio

    @ms.jit
    def construct(self, gradients: List[Tensor]):
        weight_decay = self.get_weight_decay()
        lr = self.get_lr()
        self.assignadd(self.global_step, self.global_step_increase_tensor)

        ops.assign(self.adamw_beta1_t, self.adamw_beta1_t * self.adamw_beta1)
        ops.assign(self.adamw_beta2_t, self.adamw_beta2_t * self.adamw_beta2)

        if self.is_group:
            if self.is_group_lr:
                optim_result = self.hyper_map(
                    ops.partial(
                        _muon_opt,
                        self.momentum,
                        self.adamw_beta1,
                        self.adamw_beta2,
                        self.adamw_beta1_t,
                        self.adamw_beta2_t,
                        self.adamw_eps,
                        self.nesterov,
                        self.ns_steps,
                    ),
                    lr,
                    weight_decay,
                    self._parameters,
                    self.moments1,
                    self.moments2,
                    gradients,
                    self.lr_ratio,
                    self.use_muon,
                    self.decay_flags,
                    self.optim_filter,
                )
            else:
                optim_result = self.hyper_map(
                    ops.partial(
                        _muon_opt,
                        self.momentum,
                        self.adamw_beta1,
                        self.adamw_beta2,
                        self.adamw_beta1_t,
                        self.adamw_beta2_t,
                        self.adamw_eps,
                        self.nesterov,
                        self.ns_steps,
                        lr,
                    ),
                    weight_decay,
                    self._parameters,
                    self.moments1,
                    self.moments2,
                    gradients,
                    self.lr_ratio,
                    self.use_muon,
                    self.decay_flags,
                    self.optim_filter,
                )
        else:
            optim_result = self.hyper_map(
                ops.partial(
                    _muon_opt,
                    self.momentum,
                    self.adamw_beta1,
                    self.adamw_beta2,
                    self.adamw_beta1_t,
                    self.adamw_beta2_t,
                    self.adamw_eps,
                    self.nesterov,
                    self.ns_steps,
                    lr,
                    weight_decay,
                ),
                self._parameters,
                self.moments1,
                self.moments2,
                gradients,
                self.lr_ratio,
                self.use_muon,
                self.decay_flags,
                self.optim_filter,
            )
        return optim_result
