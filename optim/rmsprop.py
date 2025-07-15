from typing import List, Tuple

import mindspore as ms
import mindspore.mint as mint
import mindspore.ops as ops
from mindspore import Parameter, ParameterTuple, Tensor
from mindspore.experimental.optim.optimizer import Optimizer

_rmsprop_opt = ops.MultitypeFuncGraph("rmsprop_opt")


@_rmsprop_opt.register(
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
)
def _update_run_op(
    alpha: float,
    eps: float,
    momentum: float,
    centered: bool,
    weight_decay: float,
    maximize: bool,
    lr: Parameter,
    param: Parameter,
    v: Parameter,
    b: Parameter,
    g_ave: Parameter,
    g: Tensor,
) -> bool:
    if maximize:
        g = mint.neg(g)

    if weight_decay > 0:
        g.add_(weight_decay * param)

    v_next = mint.lerp(mint.square(g), v, alpha)

    g_ave_next = None
    if centered:
        g_ave_next = mint.lerp(g, g_ave, alpha)
        v_next_hat = v_next - mint.square(g_ave_next)
    else:
        v_next_hat = v_next

    if momentum > 0:
        g = momentum * b + g / (mint.sqrt(v_next_hat) + eps)
    else:
        g = g / (mint.sqrt(v_next_hat) + eps)

    param.add_(-lr * g)

    ops.assign(v, v_next)
    if momentum > 0:
        ops.assign(b, g)
    if centered:
        ops.assign(g_ave, g_ave_next)
    return True


class RMSprop(Optimizer):
    """Following https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html"""

    def __init__(
        self,
        params: List[Parameter],
        lr: float = 0.01,
        alpha: float = 0.99,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        momentum: float = 0.0,
        centered: bool = False,
        maximize: bool = False,
    ) -> None:
        defaults = dict(
            lr=lr,
            momentum=momentum,
            alpha=alpha,
            eps=eps,
            centered=centered,
            weight_decay=weight_decay,
            maximize=maximize,
        )
        super(RMSprop, self).__init__(params, defaults)

        self.square_avg = self.parameters.clone("square_avg", init="zeros")
        if momentum > 0:
            self.momentum_buffer = self.parameters.clone(
                prefix="momentum_buffer", init="zeros"
            )
        else:
            self.momentum_buffer = ParameterTuple(
                [
                    Parameter([], name="momentum_buffer ." + x.name)
                    for x in self.parameters
                ]
            )

        if centered:
            self.grad_avg = self.parameters.clone("grad_avg", init="zeros")
        else:
            self.grad_avg = ParameterTuple(
                [Parameter([], name="grad_avg." + x.name) for x in self.parameters]
            )

        self.state_step = Parameter(Tensor(0, dtype=ms.int32))
        self.increase_tensor = Tensor(1, dtype=ms.int32)

    @ms.jit
    def rmsprop(
        self,
        alpha: float,
        eps: float,
        momentum: float,
        centered: bool,
        weight_decay: float,
        maximize: bool,
        lr: Parameter,
        gradients: Tuple[Tensor],
        start_id: int,
        end_id: int,
    ) -> bool:
        optim_result = self.hyper_map(
            ops.partial(
                _rmsprop_opt,
                alpha,
                eps,
                momentum,
                centered,
                weight_decay,
                maximize,
                lr,
            ),
            self.parameters[start_id:end_id],
            self.square_avg[start_id:end_id],
            self.momentum_buffer[start_id:end_id],
            self.grad_avg[start_id:end_id],
            gradients[start_id:end_id],
        )
        return optim_result

    def construct(self, gradients: Tuple[Tensor]) -> bool:
        self.state_step += self.increase_tensor

        for group_id, group in enumerate(self.param_groups):
            start_id = self.group_start_id[group_id]
            end_id = self.group_start_id[group_id + 1]

            self.rmsprop(
                group["alpha"],
                group["eps"],
                group["momentum"],
                group["centered"],
                group["weight_decay"],
                group["maximize"],
                group["lr"],
                gradients,
                start_id,
                end_id,
            )

        return True
