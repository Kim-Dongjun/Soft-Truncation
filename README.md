# Score Matching Model for Unbounded Data Score

This repo contains an official PyTorch implementation for the paper [`Score Matching Model for Unbounded Data Score`](https://arxiv.org/abs/2106.05527).

--------------------

We propose a score network for unbounded score function that improves previous works on score-based generative models. The Unbounded Diffusion Model (UDM) estimates the score function successfully. The sampled images for FFHQ 256/CelebA-HQ 256/LSUN Bedroom/LSUN Church datasets from UDM are given as below.

![schematic](figure/sample_figures_256.jpg)

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
## Results
Out model achieves the following performance on:

### [Performance on Image Generation Task](https://paperswithcode.com/sota/image-generation-on-cifar-10)

| Experimentsal Setup | NLL (BPD) | FID | IS |
|:----------|:-------:|:----------:|:----------:|
| `CIFAR10` | 3.04 | 2.33 | **10.11** |
| `ImageNet32` | **3.59** |||
| `ImageNet64` | **3.32** |||
| `CelebA64` | 1.93 | **2.78** ||
| `CelebA-HQ 256` || **7.16** |
| `STL-10 48` || **7.71** | **13.43** |

## Checkpoints

You can download the checkpoints [here](https://drive.google.com/drive/folders/1Wyk0ucFW-QDS_g1EcPm361LWWgWqJ6L_?usp=sharing)