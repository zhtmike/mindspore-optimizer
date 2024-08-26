import time
from typing import List, Tuple

import matplotlib.pyplot as plt
import mindspore as ms
import mindspore.nn as nn
import numpy as np
from mindspore import Model
from mindspore.dataset import Cifar10Dataset, Dataset
from mindspore.dataset.vision import Normalize, ToTensor, Resize
from mindspore.train.callback import Callback, LossMonitor
from mindcv.models.vit import vit_b_16_224

from optim import AdamW


class LossDrawer(Callback):
    def __init__(self):
        self.lr_records: List[float] = list()

    def on_train_step_end(self, run_context):
        cb_params = run_context.original_args()
        loss = float(np.mean(cb_params.net_outputs.asnumpy()))
        self.lr_records.append(loss)

    def on_train_end(self, run_context):
        plt.figure()
        plt.plot(self.lr_records, ".-")
        plt.grid()
        plt.xlabel("step")
        plt.ylabel("loss")
        plt.savefig("loss.jpg")



class TimeMonitor(Callback):
    def __init__(self) -> None:
        self.epoch_start_time = 0
        self.step_start_time = 0
        self.durations: List[int] = list()

    def on_train_epoch_begin(self, run_context) -> None:
        self.epoch_start_time = time.time()

    def on_train_step_begin(self, run_context) -> None:
        self.step_start_time = time.time()

    def on_train_step_end(self, run_context) -> None:
        duration = time.time() - self.step_start_time
        self.durations.append(duration)

    def on_train_epoch_end(self, run_context) -> None:
        epoch_duration = time.time() - self.epoch_start_time
        avg_time = np.mean(self.durations)
        self.durations = list()
        print(f"Total training time for single epoch: {epoch_duration:.3f} seconds")
        print(f"Average step time: {avg_time:.3f} seconds")


def create_dataset() -> Tuple[Dataset, Dataset]:
    data_path = "tests/data/cifar-10-batches-bin"

    transforms = [
        Resize(224),
        ToTensor(),
        Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], is_hwc=False),
    ]

    dataset = Cifar10Dataset(data_path, usage="train", num_samples=1000, shuffle=True)
    dataset = dataset.map(transforms, input_columns="image")
    dataset = dataset.map(lambda x: x.astype(np.int32), input_columns="label")
    dataset = dataset.batch(64, drop_remainder=True)

    val_dataset = Cifar10Dataset(
        data_path, usage="test", num_samples=100, shuffle=False
    )
    val_dataset = val_dataset.map(transforms, input_columns="image")
    val_dataset = val_dataset.map(lambda x: x.astype(np.int32), input_columns="label")
    val_dataset = val_dataset.batch(64, drop_remainder=False)
    return dataset, val_dataset


def main():
    ms.set_seed(0)
    ms.set_context(mode=ms.GRAPH_MODE)

    net = vit_b_16_224(num_classes=10)
    dataset, val_dataset = create_dataset()

    model = Model(
        net,
        loss_fn=nn.CrossEntropyLoss(),
        optimizer=AdamW(net.trainable_params()),
        metrics={"accuracy"},
    )
    model.fit(10, dataset, val_dataset, callbacks=[LossMonitor(), LossDrawer(), TimeMonitor()])

if __name__ == "__main__":
    main()
