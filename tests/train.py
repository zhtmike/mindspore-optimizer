import argparse
import time
from typing import Dict, List, Tuple, Type

import matplotlib.pyplot as plt
import mindspore as ms
import mindspore.mint as mint
import mindspore.nn as nn
import numpy as np
import tqdm
from mindcv.models.vit import VisionTransformer
from mindspore.dataset import Cifar10Dataset, Dataset
from mindspore.dataset.vision import ToTensor
from mindspore.experimental.optim.optimizer import Optimizer
from mindspore.train.metrics import Accuracy

from optim import CAME, AdaFactor, AdamW, Muon, RMSprop

SUPPORT_OPTIMIZER: Dict[str, Type[Optimizer]] = {
    "adafactor": AdaFactor,
    "adamw": AdamW,
    "rmsprop": RMSprop,
    "came": CAME,
    "muon": Muon,
}


class LossDrawer:
    def __init__(self) -> None:
        self.lr_records: List[float] = list()

    def update(self, loss: float) -> None:
        self.lr_records.append(loss)

    def draw(self) -> None:
        plt.figure()
        plt.plot(self.lr_records, ".-")
        plt.grid()
        plt.xlabel("step")
        plt.ylabel("loss")
        plt.savefig("loss.jpg")
        plt.close()


class TimeMonitor:
    def __init__(self) -> None:
        self.epoch_start_time = 0
        self.step_start_time = 0
        self.durations: List[int] = list()

    def on_train_epoch_begin(self) -> None:
        self.epoch_start_time = time.time()

    def on_train_step_begin(self) -> None:
        self.step_start_time = time.time()

    def on_train_step_end(self) -> None:
        duration = time.time() - self.step_start_time
        self.durations.append(duration)

    def on_train_epoch_end(self) -> None:
        epoch_duration = time.time() - self.epoch_start_time
        avg_time = np.mean(self.durations)
        self.durations = list()
        print(f"Total training time for single epoch: {epoch_duration:.3f} seconds")
        print(f"Average step time: {avg_time:.3f} seconds")


def create_dataset() -> Tuple[Dataset, Dataset]:
    data_path = "tests/data/cifar-10-batches-bin"

    transforms = [ToTensor()]

    dataset = Cifar10Dataset(data_path, usage="train", shuffle=True)
    dataset = dataset.map(transforms, input_columns="image")
    dataset = dataset.map(lambda x: x.astype(np.int32), input_columns="label")
    dataset = dataset.batch(512, drop_remainder=True)

    val_dataset = Cifar10Dataset(data_path, usage="test", shuffle=False)
    val_dataset = val_dataset.map(transforms, input_columns="image")
    val_dataset = val_dataset.map(lambda x: x.astype(np.int32), input_columns="label")
    val_dataset = val_dataset.batch(512, drop_remainder=False)
    return dataset, val_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Test Train", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-n",
        "--name",
        required=True,
        choices=list(SUPPORT_OPTIMIZER.keys()),
        help="optimizer name",
    )
    args = parser.parse_args()

    ms.set_seed(0)

    net = VisionTransformer(
        image_size=32,
        patch_size=4,
        embed_dim=192,
        depth=12,
        num_heads=3,
        num_classes=10,
    )
    net.construct = ms.jit(net.construct)

    dataset, val_dataset = create_dataset()

    if args.name == "muon":
        kwargs = dict(adamw_parameter_names=("cls_token", "pos_embed", "head"))
    else:
        kwargs = dict()

    net_with_loss = nn.WithLossCell(net, mint.nn.CrossEntropyLoss())

    optimizer = SUPPORT_OPTIMIZER[args.name](net.trainable_params(), **kwargs)

    loss_and_grad_fn = ms.value_and_grad(
        net_with_loss, grad_position=None, weights=optimizer.parameters
    )

    loss_drawer = LossDrawer()
    time_monitor = TimeMonitor()
    metric = Accuracy()

    for i in range(10):
        net_with_loss.set_train(True)
        time_monitor.on_train_epoch_begin()
        for j, (input_, label) in enumerate(
            dataset.create_tuple_iterator(num_epochs=1)
        ):
            time_monitor.on_train_step_begin()
            loss, grad = loss_and_grad_fn(input_, label)
            optimizer(grad)
            time_monitor.on_train_step_end()
            loss_drawer.update(loss.item())
            print(f"epoch: {i}, step: {j}, loss: {loss.item():.3f}")
        time_monitor.on_train_epoch_end()
        loss_drawer.draw()

        net_with_loss.set_train(False)
        for input_, label in tqdm.tqdm(
            val_dataset.create_tuple_iterator(num_epochs=1),
            desc="validate",
            total=len(val_dataset),
        ):
            pred = net(input_)
            metric.update(pred, label)
        accuracy = metric.eval()
        print(f"validation accuracy: {accuracy:.2f}")
        metric.clear()


if __name__ == "__main__":
    main()
