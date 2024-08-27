# (Unofficial) Mindspore Optimizer

An implementation of the Mindspore optimizer that functions similarly to PyTorch’s optimizer.

Highlights:
- support native FP16 / BF16 training, or AMP training. 
- support group learning rate / group weight decay.
- some optimizers *may offer* better speed compared with official ones. (e.g., adafactor), Feel free to experiment!

## Support Optimizers

- RMSprop
- AdamW
- AdaFactor

## Environment

- python >= 3.9
- mindspore >= 2.2.14

## Test
To test the optimizer, follow these steps using the CIFAR-10 dataset:

1. Download the CIFAR-10 binary dataset and unzip it with the following command:

```bash
wget -c https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/cifar-10-binary.tar.gz -P tests/data/ && tar xzf tests/data/cifar-10-binary.tar.gz -C tests/data/
```

2. Install `mindcv >= 0.3.0` by

```bash
pip install mindcv
```

2. Start training by running:

```bash
PYTHONPATH=$(pwd):$PYTHONPATH python tests/train.py -n adafactor
```

During training, the results will be displayed in the terminal, and an additional loss curve plot named `loss.jpg` will be saved.
