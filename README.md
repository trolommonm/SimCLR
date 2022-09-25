# PyTorch SimCLR: A Simple Framework for Contrastive Learning of Visual Representations

## Introduction

This is a fork of [sthalles/SimCLR](https://github.com/sthalles/SimCLR)'s repo and an unofficial pytorch implementation
of [SimCLR](https://arxiv.org/abs/2002.05709). In order to more closely replicate the results achieved by the authors, 
the following changes were made:
- LARS optimizer was added
- Linear Warmup (for 10 epochs) with Cosine Annealing LR scheduler was added
- For CIFAR10 dataset, I replace the first 7×7 Conv of stride 2 with 3×3 Conv of stride 1, and also removed 
the first max pooling operation. For STL, I use the 3×3 Conv of stride 1 but include the first max pooling layer.

## Results

The aim was to try and replicate the results on Appendix B.9. of the [SimCLR](https://arxiv.org/abs/2002.05709)
paper. The following experiments were conducted:

| Encoder  | Learning Rate | Weight Decay | Batch Size | Epochs | Temperature | Momentum | Color Distortion Strength | Linear Evaluation |
|----------|---------------|--------------|------------|--------|-------------|----------|---------------------------|-------------------|
| ResNet18 | 4             | 1e-6         | 1024       | 1000   | 0.5         | 0.9      | 0.5                       | 0.913             |
| ResNet50 | 4             | 1e-6         | 1024       | 1000   | 0.5         | 0.9      | 0.5                       | 0.931             |

For the linear evaluation, the weights of the base encoder were frozen and a fully connected layer was attached
to the base encoder and then trained with the CIFAR10 dataset. The linear evaluation accuracy reported in the table 
above are the top 1 test accuracy. 