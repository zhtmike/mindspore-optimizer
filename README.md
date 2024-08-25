# (Unofficial) Mindspore Optimizer

An implementation of the Mindspore optimizer that functions similarly to PyTorch’s optimizer

## Environment

python >= 3.9
mindspore >= 2.2.14

## Support Optimizers

- AdamW

## Test

1. prepare the dataset

```bash
wget https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/cifar-10-binary.tar.gz -P tests/data/
tar xzf data/cifar-10-binary.tar.gz -C tests/data/
rm -r tests/data/cifar-10-binary.tar.gz
```

2. run `python test/train.py`
