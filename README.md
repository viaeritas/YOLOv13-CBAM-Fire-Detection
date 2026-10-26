# YOLOv13-CBAM: Evaluating Attention Mechanism Placement in Contrast-Enhanced YOLO Models for Fire and Smoke Detection

This repository contains the official implementation, model configurations, and reproduction scripts for our manuscript: 
**"Evaluating Attention Mechanism Placement in Contrast-Enhanced You Only Look Once Models for Fire and Smoke Detection"**.

## 📌 Overview
This study evaluates the impact of Convolutional Block Attention Module (CBAM) placement across different stages (Backbone, Neck, and Head) of the YOLOv13 architecture under fixed Contrast Limited Adaptive Histogram Equalization (CLAHE) preprocessing for forest and land fire detection.

---

## ⚙️ Model Configurations

We provide the complete, executable model configuration YAML files for all evaluated CBAM variants:

| Configuration | Target Component | CBAM Insertion Stage / Layer Index | Configuration File |
| :--- | :--- | :--- | :--- |
| **YOLOv13s + CBAM Backbone** | Backbone | After DSC3k2 and A2C2f blocks (P2–P5 feature extraction) | [`yolov13s-cbam-backbone.yaml`](./yolov13s-cbam-backbone.yaml) |
| **YOLOv13s + CBAM Neck** | Neck | After DSC3k2 blocks in FPN/PAN multi-scale feature aggregation | [`yolov13s-cbam-neck.yaml`](./yolov13s-cbam-neck.yaml) |
| **YOLOv13s + CBAM Head** | Head | Immediately prior to the 3 Detect output heads | [`yolov13s-cbam-head.yaml`](./yolov13s-cbam-head.yaml) |

---

## 📊 Main Benchmark Results (CLAHE Preprocessed)

| Model Configuration | Precision (%) | Recall (%) | mAP50 (%) | mAP50-95 (%) | Latency (ms) | FPS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **YOLOv13s Default** | 85.38 | **81.78** | **86.90** | **57.61** | **26.9** | **37.2** |
| **YOLOv13s + CBAM Backbone** | **85.79** | 79.48 | 86.41 | 56.85 | 29.8 | 33.5 |
| **YOLOv13s + CBAM Neck** | 84.53 | 79.29 | 85.30 | 55.28 | 30.0 | 33.3 |
| **YOLOv13s + CBAM Head** | 84.30 | 79.81 | 85.74 | 55.62 | 29.1 | 34.4 |

---

## 🚀 Quick Start

### Installation
```bash
git clone [https://github.com/viaeritas/YOLOv13-CBAM-Fire-Detection.git](https://github.com/viaeritas/YOLOv13-CBAM-Fire-Detection.git)
cd YOLOv13-CBAM-Fire-Detection
pip install -r requirements.txt
