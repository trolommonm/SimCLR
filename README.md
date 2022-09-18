# PyTorch SimCLR: A Simple Framework for Contrastive Learning of Visual Representations

## Introduction

This is a fork of [sthalles/SimCLR](https://github.com/sthalles/SimCLR)'s repo and an unofficial pytorch implementation
of [SimCLR](https://arxiv.org/abs/2002.05709). In order to more closely replicate the results achieved by the authors, 
the following changes was made:
- LARS optimizer was added
- Linear Warmup (for 10 epochs) with Cosine Annealing LR scheduler was added
- For CIFAR10 dataset, I replace the first 7×7 Conv of stride 2 with 3×3 Conv of stride 1, and also removed 
the first max pooling operation. For STL, I use the 3×3 Conv of stride 1 but include a max pool.