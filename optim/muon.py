import math
from typing import List, Optional, Tuple, Union

import mindspore as ms
import mindspore.mint as mint
import mindspore.ops as ops
from mindspore import Parameter, ParameterTuple, Tensor
from mindspore.experimental.optim.optimizer import Optimizer

_muon_opt = ops.MultitypeFuncGraph("muon_opt")


@_muon_opt.register(
    "Float",
    "Float",
    "Float",
    "Float",
    "Bool",
    "Int",
    "Float",
    "Bool",
    "Tensor",
    "Tensor",
    "Tensor",
    "Tensor",
    "Tensor",
    "Tensor",
    "Float",
    "Bool",
)
def _update_run_op(
    mu: float,
    beta1: float,
    beta2: float,
    eps: float,
    nesterov: bool,
    ns_steps: int,
    weight_decay: float,
    maximize: bool,
    lr: Parameter,
    step: Parameter,
    param: Parameter,
    m: Parameter,
    v: Parameter,
    g: Tensor,
    ratio: float,
    use_muon: bool,
) -> bool:
    if maximize:
        g = mint.neg(g)

    if weight_decay > 0:
        param.add_(-lr * weight_decay * param)

    v_next = None
    if use_muon:
        # Muon branch
        m_next = mint.lerp(g, m, mu)
        if nesterov:
            g = mint.lerp(g, m_next, mu)
        else:
            g = m_next
        g = zeropower_via_newtonschulz5(g, steps=ns_steps)
        param.add_(-lr * ratio * g)
    else:
        # AdamW branch
        m_next = mint.lerp(g, m, beta1)
        v_next = mint.lerp(mint.square(g), v, beta2)
        m_hat = m_next / (1 - mint.pow(beta1, step))
        v_hat = v_next / (1 - mint.pow(beta2, step))
        g = m_hat / (mint.sqrt(v_hat) + eps)
        param.add_(-lr * g)

    ops.assign(m, m_next)
    if not use_muon:
        ops.assign(v, v_next)
    return True


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
    shape = G.shape
    dtype = G.dtype
    assert len(shape) >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    G = G.bfloat16()

    if len(shape) > 2:
        G = mint.reshape(G, (G.shape[0], -1))

    need_transpose = G.shape[0] > G.shape[1]
    if need_transpose:
        G = G.T
    # Ensure spectral norm is at most 1
    G = G / (mint.norm(G) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = G @ G.T
        B = (
            b * A + c * A @ A
        )  # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        G = a * G + B @ G

    if need_transpose:
        G = G.T

    if len(shape) > 2:
        G = mint.reshape(G, shape)

    return G.to(dtype)


class Muon(Optimizer):
    """Following https://github.com/MoonshotAI/Moonlight"""

    def __init__(
        self,
        params: List[Parameter],
        lr: Union[float, Tensor] = 0.001,
        momentum: float = 0.95,
        ns_steps: int = 5,
        adamw_betas: Tuple[float, float] = (0.9, 0.999),
        adamw_eps: float = 1e-8,
        nesterov: bool = True,
        weight_decay: float = 0.1,
        adamw_parameter_names: Optional[Tuple[str, ...]] = ("embed_tokens", "lm_head"),
        rms_scale: float = 0.2,
        *,
        maximize: bool = False
    ) -> None:
        if adamw_parameter_names is None:
            adamw_parameter_names = tuple([])

        if not isinstance(adamw_parameter_names, (tuple, list)):
            raise ValueError("`adamw_parameter_names` must be a None, tuple or list.")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            ns_steps=ns_steps,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
            nesterov=nesterov,
            weight_decay=weight_decay,
            maximize=maximize,
        )
        super(Muon, self).__init__(params, defaults)

        self.exp_avg = self.parameters.clone("exp_avg", init="zeros")
        self.use_muon = tuple(
            [
                (
                    True
                    if len(x.shape) >= 2
                    and not any([p in x.name for p in adamw_parameter_names])
                    else False
                )
                for x in self.parameters
            ]
        )
        self.exp_avg_sq = ParameterTuple(
            [
                (
                    Parameter(
                        mint.zeros(x.shape, dtype=x.dtype), name="exp_avg_sq." + x.name
                    )
                    if not use_muon
                    else Parameter([], name="exp_avg_sq." + x.name)
                )
                for x, use_muon in zip(self.parameters, self.use_muon)
            ]
        )

        self.lr_ratio = tuple(
            [
                self._cal_lr_ratio(x, use_muon, rms_scale=rms_scale)
                for x, use_muon in zip(self.parameters, self.use_muon)
            ]
        )

        self.state_step = Parameter(Tensor(0, dtype=ms.int32))
        self.increase_tensor = Tensor(1, dtype=ms.int32)

    def _cal_lr_ratio(
        self, param: Parameter, use_muon: bool, rms_scale: float = 0.2
    ) -> float:
        if not use_muon:
            return 1.0

        A, B = param.shape[:2]
        # We adjust the learning rate and weight decay based on the size of the parameter matrix
        # as describted in the paper
        adjusted_ratio = rms_scale * math.sqrt(max(A, B))
        return adjusted_ratio

    @ms.jit
    def muon(
        self,
        momentum: float,
        beta1: float,
        beta2: float,
        eps: float,
        nesterov: bool,
        ns_steps: int,
        weight_decay: float,
        maximize: bool,
        lr: Parameter,
        gradients: Tuple[Tensor, ...],
        ratio: Tuple[float, ...],
        use_muon: Tuple[bool, ...],
        start_id: int,
        end_id: int,
    ) -> bool:
        optim_result = self.hyper_map(
            ops.partial(
                _muon_opt,
                momentum,
                beta1,
                beta2,
                eps,
                nesterov,
                ns_steps,
                weight_decay,
                maximize,
                lr,
                self.state_step,
            ),
            self.parameters[start_id:end_id],
            self.exp_avg[start_id:end_id],
            self.exp_avg_sq[start_id:end_id],
            gradients[start_id:end_id],
            ratio[start_id:end_id],
            use_muon[start_id:end_id],
        )
        return optim_result

    def construct(self, gradients: Tuple[Tensor, ...]) -> bool:
        self.state_step += self.increase_tensor
        for group_id, group in enumerate(self.param_groups):
            beta1, beta2 = group["adamw_betas"]
            start_id = self.group_start_id[group_id]
            end_id = self.group_start_id[group_id + 1]

            self.muon(
                group["momentum"],
                beta1,
                beta2,
                group["adamw_eps"],
                group["nesterov"],
                group["ns_steps"],
                group["weight_decay"],
                group["maximize"],
                group["lr"],
                gradients,
                self.lr_ratio,
                self.use_muon,
                start_id,
                end_id,
            )

        return True
