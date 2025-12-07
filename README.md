# Unsupervised Brain Tumor Anomaly Detection Framework

This repository contains all the code used for **experiments, baselines, prototype implementations, and inference pipelines** for unsupervised anomaly detection on brain tumor MRI (BraTS / MEN).
Because this project evolved through many iterations, the repo currently includes **well-structured components**, **experimental code**, and **chaotic early prototypes**, all preserved for **reproducibility**.

The repository is organized into the following main folders:

```
├── data_analysis/          # Exploratory analysis, dataset stats, visualizations  
├── code/                   # Traoning experiments, prototypes, deprecated scripts  
└── inference/              # Inference pipeline + Dockerfile
```

---

## 📁 Folder Overview

### **1. `data_analysis/`**

Contains exploratory notebooks and scripts, such as:

- MRI modality and voxel-intensity analysis
- Lesion size statistics and distributions
- Patch sampling studies
- Visual sanity checks of preprocessing
- Experiments that informed the design of transforms and model architectures

These scripts are **not required for training**, but document the reasoning and experimental process.

---

### **2. `code/` — Main Clean Code**

This folder contains the **production-ready** implementation of the framework.

#### **Models**

- Vision Transformer Autoencoder (ViT-AE)
- Variational Autoencoder (VAE) version
- Patch-based encoders and decoders
- Optional ResNet hybrid encoders
- Fusion layers and architecture presets

#### **Datasets**

- BraTS / MEN loader
- Modality handling (T1, T1c, T2, FLAIR)
- Patch extraction and reconstruction
- Blosc2-compressed dataset support
- Normalization and transforms

#### **Training Logic**

- Clean `model.train()` loop
- Validation logic
- Losses: L1, SSIM, KL, VAE loss, expansion losses
- Full experiment configuration logging
- Checkpoint saving
- Weights & Biases integration

Use this folder for **new experiments**, **extending architectures**, or **reproducible training**.

Part of this code is:

> ⚠️ **Unrefactored and not intended for direct use.**

However, it is valuable for understanding:

- The evolution of the project
- Ideas and experiments that worked or failed
- Intermediate results

---

### **3. `inference/` — Production Inference + Docker**

This folder includes:

#### **Inference script**

Loads a checkpoint and generates:

- Reconstruction maps
- Anomaly heatmaps
- Patch-wise aggregation
- Optional Grad-CAM overlays

#### **Dockerfile**

A reproducible container for inference:

```bash
docker build -t anomaly_inference .
docker run --gpus all     -v /data:/input anomaly_inference     --image_path /input/subject_001.nii.gz     --output_path /input/pred_001.nii.gz
```

The container includes:

- Python 3.10
- PyTorch + CUDA
- MONAI, nibabel
- All required model dependencies

Suitable for **deployment**, **cluster inference**, or **cloud execution**.
