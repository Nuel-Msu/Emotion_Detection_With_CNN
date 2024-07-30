# Emotion_detection Using CNNs

## Introduction

This repository contains code for building a Convolutional Neural Network (CNN) model to classify facial expressions using TensorFlow and Keras. The dataset used is the Facial Expression Recognition Dataset, which consists of grayscale images categorized into 7 classes (angry, disgust, fear, happy, neutral, sad, surprise).

## Libraries Used

- TensorFlow (version >= 2.0)
- Keras (from TensorFlow.keras)
- NumPy
- Matplotlib
- PIL (Python Imaging Library)
- Python 3,
- [OpenCV](https://opencv.org/)
- [Tensorflow](https://www.tensorflow.org/)

## Dataset

The dataset used for training and validation is the Facial Expression Recognition Dataset, located in `/kaggle/input/face-expression-recognition-dataset/images`. It contains:
- 28,821 training images
- 7,066 validation images

## Installation

* Python 3, [OpenCV 3 or 4](https://opencv.org/), [Tensorflow 1 or 2](https://www.tensorflow.org/)
* To install the required packages, run `pip install -r requirements.txt`.

## Usage

The repository is currently compatible with `tensorflow-2.0` and makes use of the Keras API using the `tensorflow.keras` library.

* First, clone the repository with `git clone https://github.com/Nuel-Msu/Emotion_Detection_With_CNN.git` and enter the cloned folder: `cd Emotion_Detection_With_CNN`.

* Download the FER-2013 dataset from [here](https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset) and unzip it inside the `Tensorflow` folder. This will create the folder `data`.

* If you want to train this model or train after making changes to the model, kindly visit.[my Kaggle account](https://www.kaggle.com/code/oladeneyux/emotion-detection-dataset) where the model was originally trained

* If you want to view the predictions without training again, you can download my pre-trained model `(emotion-detect.keras)` from [here](https://www.kaggle.com/code/oladeneyux/emotion-detection-dataset/output) and then run `python Detectemotion.py`.

# Data Preprocessing
* Images are resized to 48x48 pixels and normalized. Data augmentation techniques such as shear, zoom, and horizontal flip are applied to the training set to enhance model generalization.
* The [original FER2013 dataset in Kaggle](https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset) is available as a single folder in the archieve. I had converted into a dataset of images in the JPG format for training/validation.


# Model Architecture
The CNN model architecture is defined as follows:
- Input Layer: Accepts grayscale images of size 48x48 pixels.
- Convolutional Layers: Three sets of Conv2D, BatchNormalization, Activation (ReLU), MaxPooling2D, and Dropout layers.
- Fully Connected Layers: Two Dense layers with BatchNormalization, ReLU activation, and Dropout.
- Output Layer: Dense layer with softmax activation for multi-class classification.
- The model is compiled with the Adam optimizer and categorical cross-entropy loss function.

# Results
After training for a certain number of epochs, evaluate the model's performance on the validation set to assess its accuracy. Below are the plots showing the training and validation accuracy and loss over epochs:

### Model Accuracy 

![Model Accuracy](https://github.com/user-attachments/assets/df760943-813e-4392-bf8e-e402b715c942)

## Loss Accuracy
![Loss Accuracy](https://github.com/user-attachments/assets/69a1bd3f-d74f-40cf-850e-53edef94db06)

# Example Output
![Screenshot 2024-07-04 205145](https://github.com/user-attachments/assets/9d4d7473-22cf-4ebc-8379-f847187b8a6c)
![Screenshot 2024-07-04 204939](https://github.com/user-attachments/assets/338f80cf-de70-4d48-bafa-aa097b48b027)
![Screenshot 2024-07-04 204218](https://github.com/user-attachments/assets/50ff4d8f-3e42-4327-a0e7-7a9bd12dfc81)
![Screenshot 2024-07-04 204146](https://github.com/user-attachments/assets/9825b362-6494-4a66-985b-2316f1799d5d)
![Screenshot 2024-07-04 204104](https://github.com/user-attachments/assets/bc02f035-f295-4948-bb9e-78686d738e56)
![Screenshot 2024-07-04 204043](https://github.com/user-attachments/assets/436d8c22-0871-4f08-a9d3-42cf39fe4156)



# References
- Sun, Y., Wang, X., & Tang, X. (2014). Deep Learning Face Representation from Predicting 10,000 Classes. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR '14). IEEE, Washington, DC, USA, 1891-1898. DOI:   
                  https://doi.org/10.1109/CVPR.2014.244.
- Zhang, K., Zhang, Z., Li, Z., & Qiao, Y. (2016). Joint Face Detection and Alignment Using Multitask Cascaded Convolutional Networks. IEEE Signal Processing Letters, 23(10), 1499-1503. DOI: https://doi.org/10.1109/LSP.2016.2603342.
- Taigman, Y., Yang, M., Ranzato, M., & Wolf, L. (2014). DeepFace: Closing the Gap to Human-Level Performance in Face Verification. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR '14). IEEE, Washington, DC, USA, 1701-1708. 
        DOI: https://doi.org/10.1109/CVPR.2014.220.

