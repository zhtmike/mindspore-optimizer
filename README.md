# (Unofficial) Mindspore Optimizer

An implementation of the Mindspore optimizer that functions similarly to PyTorch’s optimizer

## Environment

- python >= 3.9
- mindspore >= 2.2.14

## Support Optimizers

- AdamW

## Test
To test the optimizer, follow these steps using the CIFAR-10 dataset:

1. Download the CIFAR-10 binary dataset and unzip it with the following command:

```bash
wget -c https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/cifar-10-binary.tar.gz -P tests/data/ && tar xzf tests/data/cifar-10-binary.tar.gz -C tests/data/
```

2. Start training by running:

```bash
PYTHONPATH=$(pwd):$PYTHONPATH python tests/train.py
```

The loss curve will be saved as `loss.jpg`.
