# Bharatanatyam Mudra Recognition

A deep learning-based project for recognizing Bharatanatyam hand mudras from image data using Convolutional Neural Networks (CNNs).

## Overview

This project focuses on classifying five Bharatanatyam mudra classes:
- Trisula
- Musti
- Sikhara
- Simhamukha
- Pataka

The model is trained on a dataset of labeled mudra images and uses preprocessing, augmentation, and CNN-based training to improve recognition performance.

## Project Structure

- `BharatanatyamMudraRecognition.ipynb` – Main notebook containing data loading, preprocessing, model training, evaluation, and visualization.
- `requirements.txt` – Project dependencies
- `mudra_classifier_model.h5` – Saved trained model
- `class_distribution.png` – Class distribution chart
- `training_validation_curves.png` – Accuracy and loss curves

## Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Seaborn
- scikit-learn

## Dataset

The dataset contains images organized into folders by mudra class. Each class contains multiple images of the corresponding hand gesture.

## Workflow

1. Load the dataset from Google Drive / local directory
2. Inspect class distribution and image characteristics
3. Resize and normalize images
4. Split data into train, validation, and test sets
5. Apply image augmentation
6. Train a CNN model
7. Evaluate model performance using accuracy and loss
8. Save the trained model

## Model Details

The model uses a custom CNN architecture with:
- Convolutional layers
- Batch normalization
- Max pooling
- Dense layers
- Dropout for regularization
- Softmax output for 5-class classification


Note : I was unable to upload pickle file to the repository because of the large size of the file
