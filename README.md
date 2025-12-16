# 🖼️ Image Classification using CNN (Deep Learning)

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red.svg)
![Dataset](https://img.shields.io/badge/Dataset-CIFAR--10-green.svg)

## 📖 Overview

This project demonstrates image classification using a Convolutional Neural Network (CNN) built with TensorFlow and Keras.  
The model is trained on the CIFAR-10 dataset, a popular benchmark dataset for computer vision tasks.

The CNN automatically learns spatial features from images and classifies them into one of ten predefined object categories.

## 📂 Dataset Information

The project uses the built-in CIFAR-10 dataset available in Keras.

- Dataset Name: CIFAR-10  
- Total Images: 60,000  
  - Training Images: 50,000  
  - Testing Images: 10,000  
- Image Resolution: 32 × 32 pixels  
- Color Channels: RGB (3 channels)  
- Number of Classes: 10  

### Class Labels

Airplane, Automobile, Bird, Cat, Deer,  
Dog, Frog, Horse, Ship, Truck

## 🛠️ Technologies & Libraries

- Python – Core programming language  
- TensorFlow / Keras – Used to build and train the CNN model  
- NumPy – Numerical operations and array handling  
- Matplotlib – Visualization of images and training metrics  
- Scikit-learn – Model evaluation utilities  

## 🧠 Model Architecture

The CNN model is built using the Keras Sequential API and consists of the following layers:

1. Convolutional Layer – Feature extraction  
2. Max Pooling Layer – Spatial down-sampling  
3. Convolutional Layer – Deeper feature learning  
4. Max Pooling Layer – Further dimensionality reduction  
5. Flatten Layer – Converts feature maps to a vector  
6. Dense Layer – Fully connected neural layer  
7. Output Layer – Softmax activation for multi-class classification  

### Compilation Details

- Optimizer: Adam  
- Loss Function: Sparse Categorical Crossentropy  
- Metrics: Accuracy  

## 📊 Training & Results

- Batch Size: 32  
- Epochs: 10  
- Training/Test Split: 50,000 / 10,000  

### Performance Evaluation

The model’s accuracy and loss are plotted across training epochs.  
Final performance is evaluated using unseen test data.

### Sample Predictions

The trained model predicts class labels for test images and compares them with the actual labels.

## 🚀 How to Run

1. Clone the repository
   git clone https://github.com/YOUR_USERNAME/Image-Classification-CNN-for-CIFAR-10.git
   cd Image-Classification-CNN-for-CIFAR-10

2. Install dependencies
   pip install numpy matplotlib tensorflow scikit-learn

3. Run the notebook
   jupyter notebook Image-Classification-CNN-for-CIFAR-10.ipynb



## 🤝 Contributing

Contributions are welcome.  
Feel free to fork the repository, make improvements, and submit a pull request.
