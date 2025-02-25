# Heart Stroke Prediction Using Neural Networks
This repository is used to train and compare the applications and outputs of Neural Networks (NN) across multiple scenarios and industries. This project focuses on predicting the risk of heart stroke using a neural network model, serving as a proof-of-concept for integrating machine learning into healthcare analytics.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Motivation](#motivation)
3. [Dataset](#dataset)
4. [Data Preprocessing](#data-preprocessing)
5. [Exploratory Analysis](#exploratory-analysis)
6. [Model Architecture](#model-architecture)
7. [Results](#results)
8. [Limitations and Future Work](#limitations-and-future-work)
9. [How to Run using bash](#how-to-run-using-bash)
10. [Contact](#contact)

## Project Overview
This project employs a neural network model to predict the risk of heart stroke based on various health indicators. It aims to provide valuable insights for early diagnosis and risk management. The model achieves an accuracy of 82% and uses factors such as age, blood pressure, cholesterol levels, heart rate, and blood flow to make predictions.

## Motivation
Heart disease is a leading cause of death globally. Early identification of individuals at risk can significantly improve patient outcomes by allowing for timely medical intervention. This project explores how machine learning, specifically neural networks, can be applied to predict the likelihood of a heart stroke based on available health data.

## Dataset
The dataset consists of anonymized medical records, including the following features:
- Age: Patient's age
- Resting Blood Pressure (mm Hg)
- Cholesterol (mg/dl)
- Maximum Heart Rate (bpm)
- Blood Flow Indicators

The data was collected from publicly available health records and has undergone preliminary cleaning.

## Data Preprocessing
Before training the model, the following preprocessing steps were applied:
- Handling Missing Values: Imputation strategies were applied where necessary.
- Normalization: Continuous variables were scaled to a range of [0, 1] to improve model training.
- Data Splitting: The dataset was divided into training (80%), validation (10%), and test (10%) sets.

## Exploratory Analysis
Key insights from the dataset include:
- Age and Heart Stroke Risk: Probability increases significantly for individuals aged 65 and above.
- Resting Blood Pressure: Higher values correlate with increased heart stroke risk.
- Cholesterol Levels: Risk escalates above 240 mg/dl, peaking around 280 mg/dl.
- Maximum Heart Rate: Higher heart rates tend to lower stroke risk due to better oxygen circulation.
- Blood Flow Indicators: Poor circulation is linked to a higher probability of heart stroke, especially with elevated cholesterol levels.

## Model Architecture
The neural network was implemented using TensorFlow/Keras with the following configuration:
- Input Layer: 5 features
- Hidden Layers: 2 hidden layers with ReLU activation functions
- Output Layer: Sigmoid activation for binary classification (heart stroke risk)
- Optimizer: Adam
- Loss Function: Binary cross-entropy

Hyperparameters
- Learning Rate: 0.001
- Batch Size: 32
- Epochs: 50

## Results
- Accuracy: 82%
- Precision: High precision for predicting high-risk cases (Class 1)
- Recall: 75% for Class 1 (room for improvement)
- F1-Score: Balanced performance across both classes
The model demonstrated strong predictive power, particularly in identifying high-risk cases. However, achieving higher recall for positive cases remains a focus for future work.

## Limitations and Future Work
### Limitations
Data Size: The dataset used is relatively small, which may affect the generalizability of the model.
Feature Limitations: The current dataset does not include LDL and HDL cholesterol levels, which could enhance model accuracy.
Model Training Time: Neural networks can be time-consuming to train, especially when experimenting with hyperparameters.

### Future Work
Expand the Dataset: Incorporate more diverse and larger datasets to improve model robustness.
Feature Engineering: Add more health indicators (e.g., LDL and HDL cholesterol levels, family history) to enhance predictive performance.
Model Optimization: Experiment with advanced architectures (e.g., CNNs, RNNs) and hyperparameter tuning to improve recall and overall accuracy.

## How to Run using bash
Clone the repository:
```bash
git clone https://github.com/FelipeLeiteDS/Heart-Stroke-Risk_Neural-Networks.git
```
Install the required dependencies:
```bash
pip install pandas numpy scikit-learn tensorflow
```
Run the script:
```bash
python Heart-Stroke-Risk_Neural-Network.py
```

## Contact
For questions or collaboration opportunities, feel free to reach out:  
Name: Felipe Leite  
Email: felipe.nog.leite@gmail.com  
LinkedIn: [Felipe Leite](https://www.linkedin.com/in/felipeleiteds/)  
Portfolio: [FelipeLeite.ca](https://www.felipeleite.ca/)  
