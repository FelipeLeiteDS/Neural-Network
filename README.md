This repository is used to train and compare the applications and outputs of Neural Network on multiple scenarios and industries

1. Heart Stroke Prediction Using Neural Networks

Overview

This project employs a neural network (NN) model to predict the risk of heart stroke based on various health indicators. It aims to serve as a proof-of-concept for integrating machine learning into healthcare analytics, providing valuable insights that can help in early diagnosis and risk management. The model achieves an accuracy of 82% and uses factors such as age, blood pressure, cholesterol levels, heart rate, and blood flow to make predictions.

Table of Contents
  •	Motivation
  •	Dataset
  •	Data Preprocessing
  •	Exploratory Data Analysis
  •	Model Architecture
  •	Results
  •	Limitations and Future Work

Motivation

Heart disease is a leading cause of death globally. Early identification of individuals at risk can significantly improve patient outcomes by allowing for timely medical intervention. This project explores how machine learning, specifically neural networks, can be applied to predict the likelihood of a heart stroke based on available health data.
Dataset
  •	The dataset consists of anonymized medical records including features like: 
    o	Age: Patient's age
    o	Resting Blood Pressure (mm Hg)
    o	Cholesterol (mg/dl)
    o	Maximum Heart Rate (bpm)
    o	Blood Flow Indicators
  •	The data was collected from publicly available health records and has undergone preliminary cleaning.

Data Preprocessing

Before training the model, several preprocessing steps were taken:
  •	Handling Missing Values: Imputation strategies were applied where necessary.
  • Normalization: Continuous variables were scaled to a range of [0, 1] to improve model training.
  • Data Splitting: The dataset was divided into training (80%), validation (10%), and test (10%) sets.

Exploratory Data Analysis

Key insights from the dataset include:
  •	Age and Heart Stroke Risk: Probability increases significantly for individuals aged 65 and above.
  •	Resting Blood Pressure: Higher values correlate with increased heart stroke risk.
  •	Cholesterol Levels: Risk escalates above 240 mg/dl, peaking around 280 mg/dl.
  •	Maximum Heart Rate: Higher heart rates tend to lower stroke risk due to better oxygen circulation.
  •	Blood Flow Indicators: Poor circulation is linked to a higher probability of heart stroke, especially with elevated cholesterol levels.

Model Architecture

The neural network was implemented using [TensorFlow/Keras or PyTorch] with the following configuration:
  •	Input Layer: 5 features
  •	Hidden Layers: 2 hidden layers with ReLU activation functions
  •	Output Layer: Sigmoid activation for binary classification (heart stroke risk)
  •	Optimizer: Adam
  •	Loss Function: Binary cross-entropy

Hyperparameters
  •	Learning Rate: 0.001
  •	Batch Size: 32
  •	Epochs: 50

Results
  •	Accuracy: 82%
  •	Precision: High precision for predicting high-risk cases (Class 1)
  •	Recall: 75% for Class 1 (room for improvement)
  •	F1-Score: Balanced performance across both classes
The model demonstrated strong predictive power, particularly in identifying high-risk cases. However, achieving higher recall for positive cases remains a focus for future work.

Limitations and Future Work

Limitations
  •	Data Size: The dataset used is relatively small, which may affect the generalizability of the model.
  •	Feature Limitations: The current dataset does not include LDL and HDL cholesterol levels, which could enhance model accuracy.
  •	Model Training Time: Neural networks can be time-consuming to train, especially when experimenting with hyperparameters.

Future Work
  •	Expand the Dataset: Incorporate more diverse and larger datasets to improve model robustness.
  •	Feature Engineering: Add more health indicators (e.g., LDL and HDL cholesterol levels, family history) to enhance

2. Neural Network Prediction Model for House Prices

Project Overview

This project builds a neural network model using Keras to predict housing prices based on four primary features:
  •	Garage Cars
  •	Garage Area
  •	Overall Quality
  •	Ground Living Area
The model architecture, data preparation steps, and evaluation metrics are detailed in this README, along with guidelines for running and understanding the code.

Table of Contents

  •	Project Overview
  •	Getting Started
  •	Data Preparation
  •	Model Architecture
  •	Training and Evaluation
  •	Results and Analysis
  •	Limitations

Getting Started
Prerequisites

This project requires Python 3.7 or later and uses the following main libraries:

  •	pandas (for data loading and manipulation)
  •	keras (for building and training the neural network)
  •	numpy (for array manipulation and error calculations)

Repository Structure

data/ - contains the dataset file (Assignment 2_BUSI 651_House Prices.xls)
src/ - main script and function files for data loading, model training, and evaluation
notebooks/ - Jupyter notebooks for exploratory data analysis (EDA) and step-by-step model building
README.md - Project documentation

Data Preparation

The data file is loaded using pandas, and key variables are selected as features for the model. We split the data into an 80/20 training and testing set to validate the model’s generalizability.

Model Architecture

Our model is a neural network with the following structure:
  •	Input Layer: 4 features
  •	Hidden Layers: Two hidden layers, each with 32 nodes and ReLU activation
  •	Output Layer: A single output node for regression
  •	The ReLU activation function is selected for its effectiveness in deep learning tasks, introducing non-linearity into the network and helping capture complex patterns in the data.

Training and Evaluation

The model is trained over 100,000 epochs to achieve optimal results. This high number of epochs ensures the model has sufficient opportunity to learn from the data, although it also increases computational time.
To evaluate the model, we use the Mean Absolute Percentage Error (MAPE), which provides insight into the accuracy of predictions relative to actual sale prices.

Results and Analysis
The final model achieved a MAPE of 15.6%. Below are some insights into feature behavior based on the model’s predictions:
  •	Garage Cars: Minimal effect on sale price.
  •	Garage Area: Notable non-linear influence on sale price.
  •	Overall Quality: Strong linear relationship with sale price.
  •	Ground Living Area: Sensitive to decreases in area, with diminishing returns when increased.

Limitations

The model has certain limitations, primarily due to the limited feature set and computational constraints:
  •	Limited Feature Set: Key variables like neighborhood location are missing, which likely impacts prediction accuracy.
  •	High Computational Cost: The model’s training time is considerable, given the high number of epochs required.
  •	Complex Non-linear Patterns: Limited ability to capture some non-linear behaviors, likely due to limited data and feature interactions.
  •	Future iterations could improve the model by adding more relevant features and using advanced neural network architectures or regularization techniques to enhance performance and reduce computational time.

Contributing
This project is open for contributions. To propose changes or enhancements, please create a pull request with a detailed description of your modifications. Ensure that your code adheres to PEP 8 standards and is well-documented.
