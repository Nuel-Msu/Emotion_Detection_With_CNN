# Facial Expression Recognition using Convolutional Neural Networks

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

### Environment

To run the code, make sure you have TensorFlow and other necessary libraries installed. You can install them using pip:
bash
- pip install tensorflow matplotlib pillow
- To install the required packages, run `pip install -r requirements.txt`.


# Running the Code
```bash
git clone https://github.com/Nuel-Msu/Emotion_Detection_With_CNN.git
cd DetectEmotion
```

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

# Training
Training is performed using ImageDataGenerator.flow_from_directory to load batches of images directly from directories, enabling efficient data handling and preprocessing.

# Checkpointing and Early Stopping
Callbacks are implemented during training to save the best model checkpoint based on validation accuracy and to stop training early if no improvement is observed

# Results
After training for a certain number of epochs, evaluate the model's performance on the validation set to assess its accuracy. Below are the plots showing the training and validation accuracy and loss over epochs:

## Training and Validation Plots

### Model Accuracy 

![Model Accuracy](https://github.com/user-attachments/assets/df760943-813e-4392-bf8e-e402b715c942)

## Loss Accuracy
![Loss Accuracy](https://github.com/user-attachments/assets/69a1bd3f-d74f-40cf-850e-53edef94db06)



# References
- Sun, Y., Wang, X., & Tang, X. (2014). Deep Learning Face Representation from Predicting 10,000 Classes. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR '14). IEEE, Washington, DC, USA, 1891-1898. DOI:   
                  https://doi.org/10.1109/CVPR.2014.244.
- Zhang, K., Zhang, Z., Li, Z., & Qiao, Y. (2016). Joint Face Detection and Alignment Using Multitask Cascaded Convolutional Networks. IEEE Signal Processing Letters, 23(10), 1499-1503. DOI: https://doi.org/10.1109/LSP.2016.2603342.
- Taigman, Y., Yang, M., Ranzato, M., & Wolf, L. (2014). DeepFace: Closing the Gap to Human-Level Performance in Face Verification. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR '14). IEEE, Washington, DC, USA, 1701-1708. 
        DOI: https://doi.org/10.1109/CVPR.2014.220.

