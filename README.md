# Image Enhancement for Handwritten Recognition

> **Author:** Team Diegans (Group 13)

## 📖 Introduction

In academic settings, especially in large lecture halls, students often rely on smartphone photos of whiteboards or chalkboards to capture lecture notes. However, these images frequently suffer from visual distortions that reduce readability and hinder automated text recognition.

This project implements a practical pipeline to enhance low-quality handwritten images and recover hidden text. We target three specific types of visual distortions:

### Specular Glare: 
Caused by strong directional lighting or flash.

### Uneven Illumination: 
Resulting from shadows or poor lighting conditions.

### Motion Blur: 
Caused by camera shake or hand movement.

For each distortion, we provide image enhancement algorithms (including SHR-Net, Retinex, and Sparse Motion Kernel) and utilize a CRNN (Convolutional Recurrent Neural Network) model for final text recognition.

# 🚀 Key Features

Glare Removal: Algorithms to detect and suppress specular highlights.

Illumination Correction: Spatial and frequency domain methods to normalize lighting.

Deblurring: Advanced kernel estimation to restore sharp edges from blurred images.

OCR Integration: A complete pipeline connecting image enhancement to a CRNN model for evaluating text recognition performance.

# 🛠️ Installation

Prerequisites
```

Python 3.8+

PyTorch (tested with recent versions)

OpenCV

NumPy

Matplotlib

```
Setup

Clone the repository:

```
git clone [https://github.com/potatofried02/Diegans_ECE253.git](https://github.com/potatofried02/Diegans_ECE253.git)
cd Diegans_ECE253
```

Install dependencies:

```
pip install -r requirements.txt
```

Prepare Datasets:

This project utilizes the IMGUR-5K dataset for training.

We also include a custom "In-The-Wild" dataset for testing real-world distortions.

Note: Ensure data is placed in the ./data directory.

# 💻 Usage

To run algorithms about uneven illumination :

```
cd src/uneven_illumination
python homomorphic_filter.py
python retinex_algorithm.py
```
We can check the result in result folder.

To run the demo files utilizing the loaded checkpoint:

```
cd src
python demo.py
```


