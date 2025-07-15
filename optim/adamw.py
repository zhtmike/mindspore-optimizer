from typing import List, Tuple, Union

import mindspore as ms
import mindspore.mint as mint
import mindspore.ops as ops
from mindspore import Parameter, ParameterTuple, Tensor
from mindspore.experimental.optim.optimizer import Optimizer

_adam_opt = ops.MultitypeFuncGraph("adam_opt")


@_adam_opt.register(
    "Float",
    "Float",
    "Float",
    "Bool",
    "Float",
    "Bool",
    "Tensor",
    "Tensor",
    "Tensor",
    "Tensor",
    "Tensor",
    "Tensor",
    "Tensor",
)
def _update_run_op(
    beta1: float,
    beta2: float,
    eps: float,
    amsgrad: bool,
    weight_decay: float,
    maximize: bool,
    lr: Parameter,
    step: Parameter,
    param: Parameter,
    m: Parameter,
    v: Parameter,
    v_max: Parameter,
    g: Tensor,
) -> bool:
    if maximize:
        g = mint.neg(g)

    if weight_decay != 0:
        param.mul_(1 - lr * weight_decay)

    m_next = mint.lerp(g, m, beta1)
    v_next = mint.lerp(mint.square(g), v, beta2)

    m_hat = m_next / (1 - mint.pow(beta1, step))
    v_hat = v_next / (1 - mint.pow(beta2, step))

    v_max_hat = None
    if amsgrad:
        v_max_hat = mint.maximum(v_max, v_hat)
        g = m_hat / (mint.sqrt(v_max_hat) + eps)
    else:
        g = m_hat / (mint.sqrt(v_hat) + eps)

    param.add_(-lr * g)

    ops.assign(m, m_next)
    ops.assign(v, v_next)
    if amsgrad:
        ops.assign(v_max, v_max_hat)
    return True


class AdamW(Optimizer):
    """Following https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html"""

    def __init__(
        self,
        params: List[Parameter],
        lr: Union[float, Tensor] = 0.001,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        amsgrad: bool = False,
        *,
        maximize: bool = False
    ) -> None:
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            maximize=maximize,
        )
        super(AdamW, self).__init__(params, defaults)

        self.exp_avg = self.parameters.clone("exp_avg", init="zeros")
        self.exp_avg_sq = self.parameters.clone("exp_avg_sq", init="zeros")
        if amsgrad:
            self.max_exp_avg_sq = self.parameters.clone(
                prefix="max_exp_avg_sq", init="zeros"
            )
        else:
            self.max_exp_avg_sq = ParameterTuple(
                [
                    Parameter([], name="max_exp_avg_sq." + x.name)
                    for x in self.parameters
                ]
            )

        self.state_step = Parameter(Tensor(0, dtype=ms.int32))
        self.increase_tensor = Tensor(1, dtype=ms.int32)

    @ms.jit
    def adamw(
        self,
        beta1: float,
        beta2: float,
        eps: float,
        amsgrad: bool,
        weight_decay: float,
        maximize: bool,
        lr: Parameter,
        gradients: Tuple[Tensor, ...],
        start_id: int,
        end_id: int,
    ) -> bool:
        optim_result = self.hyper_map(
            ops.partial(
                _adam_opt,
                beta1,
                beta2,
                eps,
                amsgrad,
                weight_decay,
                maximize,
                lr,
                self.state_step,
            ),
            self.parameters[start_id:end_id],
            self.exp_avg[start_id:end_id],
            self.exp_avg_sq[start_id:end_id],
            self.max_exp_avg_sq[start_id:end_id],
            gradients[start_id:end_id],
        )
        return optim_result

    def construct(self, gradients: Tuple[Tensor, ...]) -> bool:
        self.state_step += self.increase_tensor
        for group_id, group in enumerate(self.param_groups):
            beta1, beta2 = group["betas"]
            start_id = self.group_start_id[group_id]
            end_id = self.group_start_id[group_id + 1]

            self.adamw(
                beta1,
                beta2,
                group["eps"],
                group["amsgrad"],
                group["weight_decay"],
                group["maximize"],
                group["lr"],
                gradients,
                start_id,
                end_id,
            )

        return True
