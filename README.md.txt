# Air Quality Prediction Using Machine Learning

## 📌 Project Overview

Air pollution is one of the major environmental problems affecting human health and the environment. This project uses Machine Learning techniques to analyse historical air-quality data and predict Carbon Monoxide (CO) concentration.

This project is developed as part of an S5 Information Technology Machine Learning case study.

---

## 🎯 Objectives

The main objectives of this project are:

- To analyse air-quality data.
- To perform Exploratory Data Analysis (EDA).
- To identify and handle missing values.
- To preprocess the dataset for machine learning.
- To select a suitable machine learning algorithm.
- To predict Carbon Monoxide (CO) concentration.
- To evaluate the performance of the machine learning model.
- To visualize actual and predicted values.
- To identify important features affecting air-quality prediction.

---

## 📊 Dataset

This project uses the **UCI Air Quality Dataset**.

### Dataset Information

| Property | Description |
|---|---|
| Dataset Name | Air Quality |
| Source | UCI Machine Learning Repository |
| Number of Instances | 9,358 |
| Number of Features | 15 |
| Data Type | Multivariate Time-Series |
| Sampling | Hourly |
| Target Variable | CO(GT) |

### Dataset Source

https://archive.ics.uci.edu/dataset/360/air

The dataset contains pollutant measurements, sensor readings, temperature, relative humidity and absolute humidity.

---

## 🧠 Machine Learning Algorithm

### Random Forest Regression

Random Forest Regression is used to predict the Carbon Monoxide concentration.

Random Forest was selected because:

- It can model nonlinear relationships.
- It works well with tabular datasets.
- It can handle multiple input features.
- It is relatively robust to noise.
- It provides feature importance.
- It generally performs well on structured datasets.

---

## 🔄 Project Workflow

```text
Dataset Collection
        ↓
Data Understanding
        ↓
Exploratory Data Analysis
        ↓
Missing Value Detection
        ↓
Data Preprocessing
        ↓
Feature Engineering
        ↓
Train-Test Split
        ↓
Random Forest Regression
        ↓
Prediction
        ↓
Performance Evaluation
        ↓
Visualization




## 📊 Results and Visualizations

### CO Distribution

![CO Distribution](images/co_distribution.png)

### CO Concentration Over Time

![CO Over Time](images/co_over_time.png)

### Correlation Heatmap

![Correlation Heatmap](images/correlation_heatmap.png)

### Actual vs Predicted

![Actual vs Predicted](images/actual_vs_predicted.png)

### Feature Importance

![Feature Importance](images/feature_importance.png)