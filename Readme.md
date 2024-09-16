# Handwritten Digit Recognition using SVM

## Project Overview
This project focuses on building a handwritten digit recognition system using the **Support Vector Machine (SVM)** algorithm. The dataset used is the popular **MNIST Digits Dataset**, which contains images of handwritten digits (0-9). Key steps in this project include importing necessary libraries, loading the dataset, data normalization, training an SVM classifier, and evaluating model performance through metrics such as accuracy, precision, recall, and F1 score.

## Table of Contents
1. [Installation](#installation)
2. [Dataset Overview](#dataset-overview)
3. [Data Preprocessing](#data-preprocessing)
4. [Model Training](#model-training)
5. [Evaluation Metrics](#evaluation-metrics)
6. [Predictions and Visualization](#predictions-and-visualization)
7. [Conclusion](#conclusion)

## Installation
To run this project, you'll need to install the following Python libraries:

```bash
pip install numpy pandas scikit-learn matplotlib
```

## Dataset Overview
The dataset used is the MNIST Digits Dataset from sklearn.datasets. It consists of images of handwritten digits, each represented as an 8x8 matrix of pixel values.

Data Shape: The dataset contains 1,797 images of handwritten digits.
Target Variable: The target variable represents the actual digit (0-9).
Features: Each image is represented by a matrix where each pixel corresponds to an intensity value between 0 and 16.
## Data Preprocessing
Import Libraries: We first import all the necessary libraries for data manipulation, model building, and visualization.

``` python

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics
import matplotlib.pyplot as plt
``` 
Load Dataset: Load the MNIST Digits Dataset from sklearn.datasets.

``` python

digits = datasets.load_digits()
``` 
Train-Test Split: We split the dataset into training and testing sets, using an 80-20 split.

``` python

x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=42)
``` 
Normalization: Normalize the pixel intensity values to a range between 0 and 1 by dividing by 16.

``` python

x_train = x_train / 16.0
x_test = x_test / 16.0
``` 
Normalization ensures that all pixel values are on a similar scale, which is crucial for algorithms like SVM to perform well.

## Model Training
We use Support Vector Machines (SVM) with a linear kernel to classify the digits.

Initialize and Train the Model: We start by using a linear kernel for the SVM. You can experiment with other kernels such as 'rbf', 'poly', etc.

``` python

svm_classifier = SVC(kernel='linear', C=1.0)
svm_classifier.fit(x_train, y_train)
``` 
Make Predictions: Use the trained model to make predictions on the test set.

``` python

y_pred = svm_classifier.predict(x_test)
``` 
## Evaluation Metrics
The model is evaluated using the following metrics:

Accuracy: Measures the percentage of correct predictions.
Precision: Measures the proportion of positive predictions that are correct.
Recall: Measures the proportion of actual positives that are correctly predicted.
F1 Score: Harmonic mean of precision and recall.
``` python

accuracy = metrics.accuracy_score(y_test, y_pred)
precision = metrics.precision_score(y_test, y_pred, average='weighted')
recall = metrics.recall_score(y_test, y_pred, average='weighted')
f1_score = metrics.f1_score(y_test, y_pred, average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1_score:.4f}")
``` 
Example Output:
Accuracy: 0.9861
Precision: 0.9863
Recall: 0.9861
F1 Score: 0.9861
## Predictions and Visualization
We can also visualize the predictions for some sample images from the test set.

``` python

new_digit_predictions = svm_classifier.predict(x_test[:5])

for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(x_test[i].reshape(8, 8), cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title(f'Predicted: {new_digit_predictions[i]}')
    plt.axis('off')

plt.show()
``` 
This code visualizes the first five images in the test set along with their predicted labels.

## Conclusion
This project demonstrates how to use the Support Vector Machine (SVM) algorithm to build a handwritten digit recognition model. We achieved high accuracy using the linear kernel of the SVM. Future improvements may include experimenting with other kernels, hyperparameter tuning, or using other machine learning algorithms like K-Nearest Neighbors (KNN) or Convolutional Neural Networks (CNN) for improved performance.

``` rust
This README gives an overview of the project, including instructions for installation, steps for preprocessing, model training, evaluation, and visualization.
``` 





