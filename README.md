<div align="center">

# 🚀 DRIFT-SENSE
### Precision SEM Wafer Alignment & Sub-Pixel Drift Recovery Engine

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green?style=for-the-badge&logo=opencv)](https://opencv.org/)
[![Accuracy](https://img.shields.io/badge/Pass%20%40%201px-100%25-brightgreen?style=for-the-badge)]()
[![Subpixel Precision](https://img.shields.io/badge/Mean%20Error-0.379px-success?style=for-the-badge)]()
[![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

---

**DRIFT-SENSE** is an ultra-high-precision, real-time spatial alignment and drift recovery framework designed for Semiconductor Scanning Electron Microscopy (SEM) wafer inspection systems. It eliminates spatial drift, heavy visual noise, and repeating VIA/array pattern ambiguities—delivering sub-pixel alignment under **27 ms per frame**.

</div>

---

## 📌 Table of Contents
- [✨ Key Highlights](#-key-highlights)
- [📊 Benchmark Performance](#-benchmark-performance)
- [🛠️ Mathematical Foundation & Methodology](#️-mathematical-foundation--methodology)
- [📁 Repository Architecture](#-repository-architecture)
- [⚡ Quickstart Guide](#-quickstart-guide)
- [🖥️ Interactive Web Dashboard](#️-interactive-web-dashboard)
- [📜 License](#-license)

---

## ✨ Key Highlights

> 🎯 **Flawless Sub-Pixel Precision**: Achieves an astounding **0.379 px Mean Error** and **100% Pass Rate @ 1px** across validation datasets.  
> ⚡ **Real-Time Pipeline**: Operates at **~38 FPS (26.61 ms / frame)**, comfortably below the 50 ms inline fab latency budget.  
> 🛡️ **Sub-Pixel Peak Upsampling**: Employs local 10x bicubic surface interpolation to eliminate pixel discretization noise.

---

## 📊 Benchmark Performance

Evaluated on **100 validation wafer image pairs** ($1000 \times 1000$ resolution, dynamic noise, complex chip feature shift):

| Metric | Benchmark Result | Target Requirement | Status |
| :--- | :---: | :---: | :---: |
| **Mean Error** | **0.379 px** | $< 5.0$ px | 🟢 **Sub-Pixel Precision** |
| **Median Error** | **0.399 px** | $< 5.0$ px | 🟢 **Passed** |
| **95th Percentile Error** | **0.599 px** | $< 5.0$ px | 🟢 **Passed** |
| **Pass Rate @ 5px** | **100.0%** | $> 95.0\%$ | 🟢 **Optimal** |
| **Pass Rate @ 4px** | **100.0%** | $> 95.0\%$ | 🟢 **Optimal** |
| **Pass Rate @ 2px** | **100.0%** | $> 90.0\%$ | 🟢 **Optimal** |
| **Pass Rate @ 1px** | **100.0%** | $> 80.0\%$ | 🟢 **Perfect Score** |
| **Mean Latency** | **26.61 ms / pair** | $< 50.0$ ms | ⚡ **Real-Time Execution** |

---

## 🛠️ Mathematical Foundation & Methodology

DRIFT-SENSE combines **Normalized Cross-Correlation (NCC)** with **Local ROI Bicubic Sub-Pixel Upsampling**:

1. **Spatial ROI Selection**: Extracts a central $500 \times 500$ region from the reference SEM image to preserve global feature context.
2. **Coarse Peak Search**: Computes correlation surfaces to locate integer peak location $(p_x, p_y)$.
3. **10x Local Sub-Pixel Refinement**:
   Extracts a local $5 \times 5$ correlation window around $(p_x, p_y)$ and resizes by a factor of $10\times$ using bicubic interpolation to reconstruct sub-pixel Hessian continuous surfaces:

   $$\text{Offset}_{sub} = \frac{\arg\max_{(u,v)} I_{up}(u,v) - \text{Pad} \cdot \text{Scale}}{\text{Scale}}$$

4. **Coordinate Alignment**: Calculates final recovered center coordinates:

   $$(x_{\text{pred}}, y_{\text{pred}}) = \left(p_x + \frac{W_c}{2} + \delta_x, \; p_y + \frac{H_c}{2} + \delta_y\right)$$

---

## 📁 Repository Architecture

```tree
DRIFT-SENSE-Wafer-Recovery/
├── dataset/
│   ├── train/                  # Training image pairs + labels.csv
│   └── val/                    # Validation image pairs + labels.csv
├── model/
│   └── drift_sense_model.pth    # Deep Learning backbone weights
├── results/
│   ├── predictions_manifest.csv # Full per-image evaluation results
│   └── worst_pair_18_err0.7px.png # Artifact log for worst failure case
├── src/
│   └── dl_training.py          # Siamese Neural Network training suite
├── app.py                      # Flask Interactive Web UI Dashboard
├── evaluate.py                 # Pipeline evaluation entry point
├── generate_dataset.py         # SEM Wafer pattern synthetic generator
├── inference.py                # Fast CLI single-pair localizer
├── localize.py                 # Core Sub-Pixel NCC Localization Engine
├── requirements.txt            # Project dependencies
└── README.md                   # Project Documentation
```

---

## ⚡ Quickstart Guide

### 1. Installation & Environment Setup
Clone the repository and install dependencies:
```bash
git clone [https://github.com/ashikarathore5/DRIFT-SENSE-Wafer-Recovery.git](https://github.com/ashikarathore5/DRIFT-SENSE-Wafer-Recovery.git)
cd DRIFT-SENSE-Wafer-Recovery
pip install -r requirements.txt
```

### 2. Generate Synthetic SEM Dataset
Create synthetic training and validation wafer splits:
```bash
python generate_dataset.py
```

### 3. Run Benchmark Suite
Run the full precision evaluation benchmark across all validation pairs:
```bash
python localize.py
```
> *Alternative shortcut:*
> ```bash
> python evaluate.py
> ```

### 4. Single-Pair Inference CLI
Test alignment recovery on a specific pair of reference and search images:
```bash
python inference.py --ref_path dataset/val/ref/ref_0000.png --search_path dataset/val/search/search_0000.png
```

---

## 🖥️ Interactive Web Dashboard

Launch the web GUI to visually inspect drift recovery and real-time heatmap overlays:

```bash
python app.py
```

Once running, navigate to **`http://localhost:5000`** in your browser.

---

## 📜 License
This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.