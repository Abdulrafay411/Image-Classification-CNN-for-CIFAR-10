# 🖼️ CIFAR-10 Image Classification with CNN

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 📖 Overview

This project implements a **Convolutional Neural Network (CNN)** using TensorFlow and Keras to classify images from the famous **CIFAR-10 dataset**. The model is trained to recognize 10 different categories of objects with an accuracy of approximately **73%** on the test set.

This repository demonstrates the end-to-end workflow of a computer vision project, including data preprocessing, model architecture design, training, and visualization of results.

## 📂 Dataset

The **CIFAR-10** dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. There are 50,000 training images and 10,000 test images.

**Classes:**
`Airplane`, `Automobile`, `Bird`, `Cat`, `Deer`, `Dog`, `Frog`, `Horse`, `Ship`, `Truck`.

## 🛠️ Technologies Used

* **Python**: Core programming language.
* **TensorFlow / Keras**: For building and training the neural network.
* **NumPy**: For numerical matrix operations.
* **Matplotlib**: For visualization of images and training history.

## 🧠 Model Architecture

The model is built using the Keras `Sequential` API and consists of the following layers:

1.  **Convolutional Layers**: Multiple `Conv2D` layers with `ReLU` activation to extract features (edges, textures, patterns).
2.  **Pooling Layers**: `MaxPooling2D` layers to downsample the spatial dimensions and reduce computation.
3.  **Flatten Layer**: Converts the 2D feature maps into a 1D vector.
4.  **Dense Layers**: Fully connected layers for classification.
5.  **Output Layer**: A Dense layer with 10 units and `Softmax` activation to output probabilities for the 10 classes.

```python
# Summary of the architecture used:
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
    layers.MaxPooling2D((2,2)),
    # ... multiple Conv/Pool layers ...
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])
