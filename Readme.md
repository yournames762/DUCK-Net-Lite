# DUCK-Net for Skin Lesion Segmentation

This repository contains the code and models used in our study:  
**"Applying and Adapting DUCK-Net for Skin Lesion Segmentation"**  
📄 *Authors*: Thoi-Toan Nguyen Dang et al.  
📅 *Date*: July 2025

## 📝 Abstract
We adapt DUCK-Net, a lightweight convolutional neural network originally designed for polyp segmentation, to the task of skin lesion segmentation using the ISIC-2016 dataset. We also propose two compressed variants—DUCK-Net-Lite and DUCK-Net-Tiny—to explore the accuracy-efficiency tradeoff for deployment in resource-constrained environments.

Our best model achieves:
- **Dice Coefficient**: 92.52%
- **IoU**: 86.08%
- Trained **from scratch**, without pretrained weights.

## 📊 Key Contributions
- ✅ Adapted DUCK-Net to ISIC-2016 without pretrained weights.
- 🪶 Proposed DUCK-Net-Lite with 45% fewer parameters and minimal accuracy loss.
- 📉 Conducted ablation studies on block structure and depth.
- ⚙️ Implemented with TensorFlow 2.11 and Albumentations for online augmentation.


## 📂 Project Structure
```
📁 DUCK-Net-Lite/             # Main implementation of DUCK-Net-Lite
📁 DUCK-Net-Tiny/             # Depth-reduced version (DUCK-Net-Tiny)
📁 DUCK-Net-Ablation-Study/   # Experiments replacing DUCK blocks with Conv
📁 Paper                      # Contains the full PDF paper
├── README.md                  # Project documentation
```


## 📄 Paper Access

🧾 [Full Paper (PDF)](https://github.com/yournames762/Lite/blob/main/Paper/Applying%20and%20Adapting%20DUCK-Net%20for%20Skin%20Lesion%20Segmentation.pdf)

> ⚠️ **Note:** This paper is a working draft submitted for review. It has not been peer-reviewed or officially published yet. Please do not cite or redistribute without permission.

## 📌 Citation

This work builds on [DUCK-Net](https://doi.org/10.1038/s41598-023-36940-5) proposed by Dumitru et al. (2023), originally designed for polyp segmentation. We adapt and compress the architecture for skin lesion segmentation on ISIC-2016.
