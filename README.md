Heart Stroke Prediction Using Neural Networks

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
