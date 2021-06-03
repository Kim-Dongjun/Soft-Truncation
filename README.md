# Score Matching Model for Unbounded Data Score

This repo contains an official PyTorch implementation for the paper `Score Matching Model for Unbounded Data Score`.

--------------------

We propose a score network for unbounded score function that improves previous work on score-based generative models. The Unbounded Noise Conditional Score Network (UNCSN) estimates the score function while pertaining the local Lipschitzness. The sampled images for FFHQ 256/CelebA-HQ 256/LSUN Bedroom/LSUN Church datasets from UNCSN are given as below.

![schematic](figure/sample_figures_256.jpg)

We achieved a negative log-likelihood (bits per dim) of [**2.06**](https://paperswithcode.com/sota/image-generation-on-cifar-10) (SOTA), a FID of **2.33**, and an Inception Score of [**10.11**](https://paperswithcode.com/sota/image-generation-on-cifar-10) (SOTA) on CIFAR10. In addition, we achieved a FID of [**7.16**](https://paperswithcode.com/sota/image-generation-on-celeba-hq-256x256) (SOTA) on CelebA-HQ 256.

## How to run the code

### Dependencies

The following dependencies are required to run the code.
```
ml-collections==0.1.0
tensorflow-gan==2.0.0
tensorflow_io
tensorflow_datasets==3.1.0
tensorflow==2.4.0
tensorflow-addons==0.12.0
tensorboard==2.4.0
absl-py==0.10.0
torch>=1.7.0
torchvision
imageio
ninja
```
### Usage

Train and evaluate our models through `main.py`.

```sh
main.py:
  --config: python file path for the training configuration
    (default: 'None')
  --assetdir: The folder name of assets for the stats file
  --eval_folder: The folder name for storing evaluation results such as samples
    (default: 'eval')
  --mode: <train|eval> eval mode contains likelihood evaluation and FID/IS computation
  --workdir: Working directory to save all artifacts such as checkpoints/samples/log
```
## Performances
| Experimentsal Setup | NLL (BPD) | FID-50k | IS-50k |
|:----------|:-------:|:----------:|:----------:|
| `cifar10_uncsn_1e-3/` | 2.96 | 2.55 | 9.97 |
| `cifar10_uncsn_deep_1e-3_mid/` | 2.83 | **2.33** | **10.11** |
| `cifar10_uncsn_deep_1e-5_mid/` | **2.06** | 2.58 | 9.74 |
| `cifar10_uncsn_deep_1e-5/` | 2.35 | 2.38 | 9.87 |
