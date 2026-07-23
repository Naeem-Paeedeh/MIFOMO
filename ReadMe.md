# MIFOMO

This implementation of the MIFOMO algorithm from the **Cross-Domain Few-Shot Learning for Hyperspectral Image Classification Based on Mixup Foundation Model** paper.

## Tested with the following packages

- numpy version: 1.26.4
- Python version: 3.12.2
- PyTorch version: 2.9.1+cu128
- TorchVision version: 0.24.1+cu128

## Model

Download the pretrained ViT-B backbone weights for both branches from the [HyperSIGMA repository](https://github.com/WHU-Sigma/HyperSIGMA).

## Datasets

You can download the datasets from the following links:

- [Chikusei dataset](https://naotoyokoya.com/Download.html)
- [Indian Pines, Salinas, and Pavia University](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes)
- [Houston dataset from the CASCL repositoty](https://github.com/Li-ZK/CDFS-CASCL-2024/)
- [Kennedy Space Center (KSC)](https://rslab.ut.ac.ir/data)

## How to run

Set the directories in the JSON files in the configs directory. Next, simply use the bash scripts in the main directory.

## Acknowledgement

- [FDFSL](https://github.com/Qba-heu/FDFSL)
- [CDFS-CASCL](https://github.com/Li-ZK/CDFS-CASCL-2024/).
