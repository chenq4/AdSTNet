# AdSTNet

Official Pytorch Code base for [Adaptive Scale Thresholding Made Efficient: AdSTNet for Blurred Lesion Image Segmentation]

## Introduction
Medical image segmentation is crucial for disease diagnosis, as precise results aid clinicians in locating lesion regions. However, lesions often have blurred boundaries and complex shapes, challenging traditional methods in capturing clear edges, which impacts accurate localization and complete excision. Small lesions are also critical but prone to detail loss during downsampling, reducing segmentation accuracy. To address these issues, we propose the Adaptive Scale Thresholding Network (AdSTNet), an innovative post-processing network that enhances sensitivity to lesion edges and cores through a dual-threshold adaptive mechanism. The dual-threshold adaptive mechanism is a key architectural component including a main threshold map for core localization and an edge threshold map for clearer boundary detection. AdSTNet is compatible with any segmentation network and is applied only during training to avoid inference-time costs. Additionally, Spatial Attention and Channel Attention (SACA), the Laplacian operator, and the Fusion Enhancement module are introduced to improve feature processing. SACA enhances spatial and channel attention for core localization; the Laplacian operator retains edge details without added complexity; and the Fusion Enhancement module adapts concatenation operation and ConvGLU to improve feature intensities. Experiments show that AdSTNet achieves notable performance gains on BUSI, ISIC, and Kvasir-SEG datasets. Most models generally process an image in around 1 second, offering reliable support for clinical diagnosis in complex scenarios.

## 1. Environment
- Please prepare an environment with Python 3.10 and PyTorch 2.1.2.

- Clone this repository:

```bash
git clone https://github.com/chenq4/AdSTNet
cd AdSTNet
```

To install all the dependencies using pip:
```bash
pip install -r requirements.txt
conda activate AdSTNet
```
## 2. Preprocess

View file `preprocess.ipynb`

## 3. Training and Validation

The results of training, testing, and visualization are all in the `train_AdptiveSAdaptiveScale_with_unet.ipynb` file

