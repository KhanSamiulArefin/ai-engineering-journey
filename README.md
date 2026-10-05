# Customer Intelligence System

## Overview

A machine learning system that predicts customer churn using customer behavioral and demographic data.

The project demonstrates an end-to-end machine learning workflow including data preprocessing, model training, evaluation, comparison, hyperparameter optimization, and model persistence.

---

## Problem Statement

Customer retention is important for businesses. This system predicts whether a customer is likely to churn based on:

- Age
- Income
- Purchase behavior
- Visit frequency

---

## Machine Learning Workflow
Customer Data
      |
      ↓
Data Exploration
      |
      ↓
Feature Engineering
      |
      ↓
Train/Test Split
      |
      ↓
Model Training
      |
      ↓
Model Evaluation
      |
      ↓
Model Optimization
      |
      ↓
Saved Model


---

## Models Implemented

### 1. Logistic Regression

Used as baseline model.

### 2. Decision Tree

Captures nonlinear decision patterns.

### 3. Random Forest

Ensemble model reducing overfitting.

### 4. XGBoost

Gradient boosting model for improved performance.

---

## Evaluation Metrics

Models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

---

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Seaborn
- Git/GitHub

---

## Future Improvements

- Deploy model using FastAPI
- Add PostgreSQL database
- Create customer dashboard
- Deploy on cloud