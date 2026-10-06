# CMSNet: Cross-Modal Affinity Learning for Medical Image Diagnosis with Knowledge-Guided Adaptive Fusion

This repository provides the official implementation of CMSNet, a cross-modal synergy network for multimodal medical image diagnosis. T

It is designed to improve cross-modal feature interaction for medical image recognition and clinical transparency for diagnostic decision-making through three key components:
1. **Feature Affinity Attention (FAA)** for modeling higher-order cross-modal feature affinity from intra-modal attention.
2. **Knowledge-guided Adaptive Fusion (KAF)** for progressively fusing multimodal features under the guidance of an evolving knowledge state.
3. **Dual-Reasoning Head (DRHead)** for combining rule-based reasoning with vision perception.

<p align="center">
  <img src="figures/framework.png" width="100%">
</p>

**Paper**: Cross-Modal Affinity Learning for Medical Image Diagnosis with Knowledge-Guided Adaptive Fusion

**Authors**: Xiaole Zhao, Wenjia Yang, Jinghui Yang, Zhenyu Wu, Ao Luo, Yan Yang, Hua Ai

---

## ✨ Overview

Multimodal medical images provide complementary information for diagnosis, but effectively exploiting the interaction between different modalities remains challenging. Existing approaches often perform independent feature extraction followed by feature fusion, which may fail to capture fine-grained cross-modal relationships. 

CMSNet addresses this problem through a progressive cross-modal reasoning framework. Given paired medical images from different modalities, CMSNet contains two modality-specific branches and a cross-modal fusion branch. The network is composed of multiple stages, with each stage containing an FAA module and a KAF module.

---

## 🔍 Method

### 1. Feature Affinity Attention (FAA)

FAA models the affinity between different modalities based on their intra-modal attention structures.

Unlike conventional co-attention mechanisms that directly construct a shared cross-modal correlation matrix, FAA derives cross-modal affinity from modality-specific attention maps. This enables higher-order interaction between multimodal features.

### 2. Knowledge-guided Adaptive Fusion (KAF)

KAF progressively integrates: 1) the current clinical feature, 2) the current dermoscopic feature, and 3) the fused feature from the previous stage. The fusion process is guided by a latent knowledge state that evolves throughout the network.

The knowledge state provides top-down guidance for feature fusion, while the resulting fused representation is used to update the knowledge state, forming a closed-loop collaborative mechanism.

### 3. Dual-Reasoning Head (DRHead)

DRHead combines two complementary reasoning pathways:

- **Rule-based Pathway (RP)**: incorporates structured diagnostic criteria.
- **Vision-based Pathway (VP)**: performs diagnosis directly from learned visual representations.

For the SPC dataset, the rule-based pathway is explicitly grounded in the seven-point checklist annotations. For datasets without such auxiliary annotations, including BraTS2019 and AMD, the corresponding pathway is learned as a latent concept pathway.

---

## 📊 Supported Datasets

CMSNet is evaluated on three public multimodal medical image datasets.

| Dataset | TasK | Modalities | Evaluation |
|---------|------|------------|------------|
|   SPC   | Skin lesion diagnosis | Clinical + Dermoscopic images | Official train/validation/test split |
|BraTS2019| Brain tumor grading | T1ce + FLAIR | 5-fold cross-validation |
|   AMD   | Macular degeneration classification | CFP + OCT | Official splitA |

---

## ⚙️ Installation

### Requirements

The implementation is developed with:
- Python 3.8+
- PyTorch 2.9.0
- compatible CUDA
- NVIDIA RTX 3090 GPU

### Setup
Clone this repository:
```
git clone https://github.com/wjyang643/CMSNet.git

cd CMSNet
```
Create the environment:
```
conda create -n cmsnet python=3.x

conda activate cmsnet
```
Install the required packages:
```bash
pip install torch torchvision
bash require.sh
pip install pandas numpy tqdm matplotlib scikit-learn
```
---

## 📁 Repository Structure
```
CMSNet/
├── main.py           # training
├── test.py           # testing
├── model.py          # network structures
├── dataloader.py     # data loading and augmentation
├── dependency.py     # Global configuration (paths, hyperparameters, etc.
├── evaluate.py       # evaluation matrics
├── require.sh        # dependencies
└── release_v0/       # dataset splitting and metadata
```
Both training and testing are controlled by configurations in `dependency.py`, without additional command-line parameters.

---

## 📈 Results
CMSNet achieves an average accuracy of 76.2% across the eight SPC tasks, including the seven-point inspection tasks and the primary diagnosis task.
<p align="center">
  <img src="figures/result1.png" width="100%">
</p>

Performance comparison of our CMSNet with more advanced backbones on SPC:
<p align="center">
  <img src="figures/result2.png" width="100%">
</p>

---

## 📦 Pretrained Models
Pretrained checkpoints will be provided here:
| Dataset |  Model |     Download  |
|---------|--------|---------------|
|    SPC  | CMSNet | [Coming soon] |
|BraTS2019| CMSNet | [Coming soon] |
|   AMD   | CMSNet | [Coming soon] |

---

## 🙏 Acknowledgements
We thank the authors of the publicly available datasets and related open-source projects used in this work.

---

## 📬 Contact
For questions, discussions, or potential collaborations, please open an issue in this repository or contact the authors.
Repository: https://github.com/zxlation/CMSNet





