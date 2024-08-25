from typing import List, Tuple
import time

import matplotlib.pyplot as plt
import mindspore as ms
import mindspore.nn as nn
import numpy as np
from mindspore import Model
from mindspore.dataset import Cifar10Dataset, Dataset
from mindspore.dataset.vision import Normalize, ToTensor
from mindspore.train.callback import Callback, LossMonitor
from net import resnet50
from optim.adamw import AdamW


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


def create_dataset() -> Tuple[Dataset, Dataset]:
    data_path = "tests/data/cifar-10-batches-bin"

    transforms = [
        ToTensor(),
        Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], is_hwc=False),
    ]

    dataset = Cifar10Dataset(data_path, usage="train", num_samples=10000)
    dataset = dataset.map(transforms, input_columns="image")
    dataset = dataset.map(lambda x: x.astype(np.int32), input_columns="label")
    dataset = dataset.batch(256, drop_remainder=True)

    val_dataset = Cifar10Dataset(data_path, usage="test", num_samples=2500)
    val_dataset = val_dataset.map(transforms, input_columns="image")
    val_dataset = val_dataset.map(lambda x: x.astype(np.int32), input_columns="label")
    val_dataset = val_dataset.batch(256, drop_remainder=False)
    return dataset, val_dataset


def main():
    ms.set_seed(0)
    ms.set_context(mode=ms.GRAPH_MODE)

    net = resnet50(10)
    dataset, val_dataset = create_dataset()

    model = Model(
        net,
        loss_fn=nn.CrossEntropyLoss(),
        optimizer=AdamW(net.trainable_params(), lr=0.0001, weight_decay=0.01),
        metrics=dict(accuracy=nn.Top1CategoricalAccuracy()),
    )
    start = time.time()
    model.fit(5, dataset, val_dataset, callbacks=[LossMonitor(), LossDrawer()])
    duration = time.time() - start
    print(f"Time Taken: {duration:.3f} seconds.")


if __name__ == "__main__":
    main()
