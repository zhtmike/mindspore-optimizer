# (Unofficial) Mindspore Optimizer

## Support Optimizers

- [RMSprop](https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
- [AdamW](https://arxiv.org/abs/1711.05101)
- [Muon](https://kellerjordan.github.io/posts/muon/)

## Environment

- python >= 3.9
- mindspore >= 2.6.0

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
PYTHONPATH=$(pwd):$PYTHONPATH python tests/train.py -n adamw
```

During training, the results will be displayed in the terminal, and an additional loss curve plot named `loss.jpg` will be saved.
