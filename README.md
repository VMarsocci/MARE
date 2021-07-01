# MARE: self-supervised Multi-Attention REsu-net for semantic segmentation in remote sensing

Scene understanding of satellite and aerial images is a pivotal task in various remote sensing (RS) applications, such as land cover and urban development monitoring. In recent years, neural networks, especially convolutional neural networks (CNNs), have become a de-facto standard in many of these applications, including object detection and disparity map creation, commonly reaching state-of-the-art performance. However, semantic segmentation still remains a challenging task. With respect to other computer vision (CV) areas, in which large repositories of annotated image data can be easily found and processed, in RS large labeled datasets are not very often available, due to their large cost and to the required manpower. On the other hand, self-supervised learning (SSL) is earning more and more interest in CV, reaching state-of-the-art in several tasks. In spite of this, most SSL models, pretrained on huge datasets like ImageNet, do not perform particularly well on RS data, due to their own peculiarities. For this reason, we propose a combination of a SSL algorithm (particularly, Online Bag of Words) and a semantic segmentation algorithm, shaped for aerial images (namely, MAResU-Net), to show new encouraging results on the ISPRS Vaihingen benchmark dataset.

<div align="center"><img src="images/cfr_mask.png", width="500"></div>

## Contents
1. [Features](#features)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Pretrained Model](#output)
5. [License](#license)
<!-- 6. [Citation](#citation) -->

## Features

MARE shows new encouraging results on the ISPRS Vaihingen benchmark dataset.

[OBOW](https://github.com/valeoai/obow/tree/main/obow)

<div align="center"><img src="images/visual_words.png", width="500"></div>

[MAResU-Net](https://github.com/lironui/MAResU-Net)

<div align="center"><img src="images/masks.png", width="500"></div>

## Installation

- Download [Python 3](https://www.python.org/)
- Install the packages:
```bash
pip install -r requirements.txt
```

## Usage 

After cloning the repo, 
modifica lo YAML (guarda esempio) e lancia il train.py

```bash
python train.py --config="Op1"
```


## Pretrained Model

At this [link](https://drive.google.com/file/d/18Ty0aUDflPjvONNszzo14JI0GjP1Rbed/view?usp=sharing), the pretrained model is available.

## License

This is an open access article distributed under the [Creative Commons Attribution License](https://creativecommons.org/licenses/by/4.0/) which permits unrestricted use, distribution, and reproduction in any medium, provided the original work is properly cited.

<!-- ## Citation

```bash
to do
``` -->
