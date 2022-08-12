# Distributed Arcface/Adaface Training in Pytorch

Modified old version of PartialFC to work with AdaFace for both normal and distributed training.
- [Killing Two Birds With One Stone: Efficient and Robust Training of Face Recognition CNNs by Partial FC (CVPR-2022)](https://arxiv.org/abs/2203.15565)
- [AdaFace: Quality Adaptive Margin for Face Recognition (CVPR-2022)](https://arxiv.org/abs/2204.00964)

## How to Training

To train a model, run `train.py` with the path to the configs:

### 1. Single node, 8 GPUs:

```shell
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=1234 train.py configs/ms1mv3_r50-adaface.py
```

### 2. Multiple nodes, each node 8 GPUs (not tested with AdaFace):

Node 0:

```shell
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr="ip1" --master_port=1234 train.py train.py configs/ms1mv3_r50-adaface.py
```

Node 1:

```shell
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr="ip1" --master_port=1234 train.py train.py configs/ms1mv3_r50-adaface.py
```

## To do
- [ ] Add ViT models
- [ ] Report comparision between Adaface, Arcface & Cosface (currently in training, dataset 5M ids and 100M images)
- [ ] Result for common large scale face recognition dataset (MS1MV2, MS1MV3, Glint360k, WebFace)

## Reference
- [Insightface](https://github.com/deepinsight/insightface) 
- [Adaface](https://github.com/mk-minchul/AdaFace)
