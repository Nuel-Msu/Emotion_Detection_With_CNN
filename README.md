# Facial Expression Recognition using Convolutional Neural Networks

This repository contains code for building a Convolutional Neural Network (CNN) model to classify facial expressions using TensorFlow and Keras. The dataset used is the Facial Expression Recognition Dataset, which consists of grayscale images categorized into 7 classes (angry, disgust, fear, happy, neutral, sad, surprise).

## Libraries Used

- TensorFlow (version >= 2.0)
- Keras (from TensorFlow.keras)
- NumPy
- Matplotlib
- PIL (Python Imaging Library)

## Dataset

The dataset used for training and validation is the Facial Expression Recognition Dataset, located in `/kaggle/input/face-expression-recognition-dataset/images`. It contains:
- 28,821 training images
- 7,066 validation images

## Setup

### Environment

To run the code, make sure you have TensorFlow and other necessary libraries installed. You can install them using pip:

```bash
pip install tensorflow matplotlib pillow
