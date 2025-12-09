##### A Deep Learning project that classifies handwritten digits (0–9) using a Convolutional Neural Network (CNN) built with TensorFlow/Keras.The model is trained on the MNIST dataset and also supports predicting digits from custom handwritten images by providing the image path.

## Live Web App:
https://mnist-digit-classification-priya.onrender.com

## Project Overview
Built a CNN model for recognizing handwritten digits.

Utilized the MNIST dataset (60,000 training and 10,000 testing images).

Achieved high accuracy(96.7%) through effective training and evaluation.

Added functionality to predict on user-provided handwritten images.

## Dataset Information
- Source: MNIST Handwritten Digit Dataset
- Training Samples: 60,000
- Test Samples: 10,000
- Image Size: 28x28 grayscale

## Technologies Used
- Python
- TensorFlow / Keras → Model building & training
- NumPy, Pandas → Data handling
- Matplotlib → Visualization
## Model Architecture
- Input Layer: 28×28 grayscale image
- Convolution Layers: For feature extraction
- Pooling Layers: For dimensionality reduction
- Dense Layers: For classification
- Output Layer: 10 neurons (digits 0–9) with softmax activation

## Achieved Accuracy
- Training Accuracy: ~95%
- Test Accuracy: ~96.71%
- Robust model performance on unseen handwritten digit images.

 ## Features
 - End-to-End CNN model for digit classification
 - Prediction on custom handwritten images (by providing image path)
 - Clean modular code for easy understanding and extension
## Project Workflow
- Import MNIST dataset
- Preprocess images (Normalization, Reshaping)
- Build CNN model
- Train model
- Evaluate accuracy
- Save trained model (.h5 / .pkl)
- Predict digits from new images

## Project Structure
 MNIST Digit Classification
├── mnist_model.ipynb       # Model training notebook
├── model.h5                # Saved trained model
├── app.py                  # Deployment backend (optional)
├── index.html              # UI (optional)
├── requirements.txt        # Dependencies
└── README.md               # Project Documentation

## Model Output

- The model predicts the handwritten digit & displays:
- Actual Digit
- Predicted Digit
- Probability Score
## Clone the repository
git clone: https://github.com/Priyyya21/MNIST-Digit-Classification
##  Use Cases
- Digit Recognition in bank checks
- Postal code automation
- OCR (Optical Character Recognition)
- Educational AI applications

##  Disclaimer

This project is for educational and research purposes only.

## Author
 Priya Sharma
 BTech CSE 3rd Year student
 Email: priyaax21@gmail.com
 Linkedin: https://www.linkedin.com/in/priya-vashishth-1790512b2/
