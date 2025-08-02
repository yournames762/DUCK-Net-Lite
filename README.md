# DUCK-Net for Skin Lesion Segmentation

This repository contains the code and models used in our study:  
**"Applying and Adapting DUCK-Net for Skin Lesion Segmentation"**  
📄 *Authors*: Thoi-Toan Nguyen Dang
📅 *Date*: August 2025

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

## 🖼️ Qualitative Results
![segmentation examples](assets/sample_results.png)  
*From top to bottom: easy, medium, and difficult cases.*

## 📂 Project Structure

