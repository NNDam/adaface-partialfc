# Distributed Arcface/Adaface Training in Pytorch

Modified old version of PartialFC to work with AdaFace for both normal and distributed training.
- [Killing Two Birds With One Stone: Efficient and Robust Training of Face Recognition CNNs by Partial FC (CVPR-2022)](https://arxiv.org/abs/2203.15565)
- [AdaFace: Quality Adaptive Margin for Face Recognition (CVPR-2022)](https://arxiv.org/abs/2204.00964)

## Benchmark

### 1. CASIA-Webface (10k ids, 0.5M images)
Current have opposite view between feature norm & image quality than described in paper
- **Accuracy**

|  Model  | Backbone | Sample Rate | LFW               | CFP-FP            | AGEDB-30          | LFW Blur          | CFP-FP Blur       | AGEDB-30 Blur     | Average    |
|:-------:|:--------:|:-----------:|-------------------|-------------------|-------------------|-------------------|-------------------|-------------------|------------|
| Arcface |   IR50   |     1.0     |         0.9920    | **0.9601** |         0.9365    |         0.9323    |         0.8517    |         0.8357    | 0.9181     |
| Adaface |   IR50   |     1.0     | **0.9923** |         0.9587    | **0.9405** | **0.9563** | **0.8667** | **0.8632** | **0.9296** |
| Arcface |   IR50   |     0.3     | **0.9923** |         0.9596    |         0.9390    |         0.9323    |         0.8480    |         0.8325    | 0.9173     |
| Adaface |   IR50   |     0.3     |         0.9915    |         0.9567    |         0.9362    |         0.9532    |         0.8570    |         0.8548    | 0.9249     |

- **Features Norm**

|  Model  | Backbone | Sample Rate | LFW          | CFP-FP       | AGEDB-30     | LFW Blur     | CFP-FP Blur  | AGEDB-30 Blur |
|:-------:|:--------:|:-----------:|--------------|--------------|--------------|--------------|--------------|---------------|
| Arcface |   IR50   |     1.0     |     12.72    |     12.97    |     13.23    |     12.56    |     13.16    |      12.86    |
| Adaface |   IR50   |     1.0     |      5.34    |       7.9    |      5.91    |     10.36    |     42.41    |      10.99    |
| Arcface |   IR50   |     0.3     |      14.3    |      14.1    |     14.54    |        14    |     14.68    |      14.29    |
| Adaface |   IR50   |     0.3     |      6.09    |     10.23    |      6.74    |      10.7    |     47.28    |       11.7    |

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
