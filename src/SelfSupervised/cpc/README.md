# CPC (simplified)

Oord et al., *Representation Learning with Contrastive Predictive Coding* (2018).

Simplified image CPC: 256x256 inputs, 7x7 overlapping patch grid, small ResNet patch encoder, GRU row context, InfoNCE on predicted future row embeddings.

Self-supervised pretrain:

```bash
python -m src.SelfSupervised.cpc.pretrain --data-dir data/imagenet100 --out-dir runs/cpc
```

Linear probe (frozen patch encoder, 256px ImageNet-100 loaders):

```bash
python -m src.SelfSupervised.cpc.linear_probe --pretrained runs/cpc/cpc_pretrained.pt --out-dir runs/cpc_probe --data-dir data/imagenet100
```
