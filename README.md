# AutoML for Healthcare Risk Prediction

## Project Overview

This project is part of the M.Tech Dissertation for the Data Science \& Engineering program at BITS Pilani (WILP).
The objective is to design and evaluate an Automated Machine Learning (AutoML) pipeline for predicting healthcare risks such as stroke, and to compare its performance with manually developed machine learning models.

## Problem Statement

Healthcare organizations generate large volumes of patient data, but building accurate predictive models requires significant manual effort and domain expertise.
This project aims to automate model development using AutoML while ensuring transparency through explainable AI techniques.

## Dataset

* Source: Kaggle – Stroke Prediction Dataset
* Problem Type: Binary Classification
* Target Variable: stroke (0 = No Stroke, 1 = Stroke)

The dataset contains demographic and clinical attributes such as age, gender, hypertension, heart disease, BMI, and average glucose level.

## Project Phases

1. Dataset Selection \& Exploratory Data Analysis (EDA)
2. Data Preprocessing \& Feature Engineering
3. Manual Machine Learning Model Development
4. AutoML Implementation and Comparison
5. Explainability \& Fairness Analysis
6. Model Deployment and Report Preparation

## Tools \& Technologies

* Python, Pandas, NumPy
* Scikit-learn
* AutoML (PyCaret / H2O AutoML)
* SHAP for Explainability
* Tableau for EDA
* Streamlit for Deployment

## Current Status

Mid-semester objectives completed.  
Project is ready for AutoML implementation, explainability analysis, and deployment in the next phase.

## Mid-Semester Progress (Completed)

The following milestones have been successfully completed as part of the mid-semester evaluation:

- Dataset selection and justification (public healthcare stroke dataset)
- Exploratory Data Analysis (EDA) using Python and Tableau
- Handling missing values and data cleaning
- Categorical feature encoding and feature scaling
- Train–test split with stratification to handle class imbalance
- Manual baseline machine learning model (Logistic Regression)
- Model evaluation using Accuracy, Confusion Matrix, and ROC-AUC
- Tableau dashboard creation for visual data exploration
- Versioned storage of cleaned and ML-ready datasets

The baseline model serves as a reference for comparison with AutoML models in the next phase of the project.

## Mid-Sem & Review Enhancements

The following enhancements were added after mid-semester evaluation based on reviewer feedback:

- Introduced a derived clinical feature (`bmi_category`) using standard BMI thresholds to improve interpretability.
- Implemented a comparative analysis using a Random Forest classifier alongside Logistic Regression.
- Evaluated both models using consistent metrics including Accuracy, Precision, Recall, F1-score, and ROC-AUC.
- Observed that Logistic Regression achieved comparable or better ROC-AUC, highlighting its suitability for healthcare risk prediction.

These updates strengthen the experimental validation and will be discussed in detail in the final dissertation report.

## AutoML Implementation (PyCaret)

AutoML was implemented using PyCaret (classification module) to automatically evaluate multiple machine learning models for stroke risk prediction. The pipeline included automated preprocessing, class imbalance handling, cross-validation, and model comparison based on ROC–AUC.
Linear Discriminant Analysis (LDA) emerged as the top AutoML model. AutoML was primarily used for benchmarking and validating the manually developed Logistic Regression model.

## Deployment & Usage (Streamlit)

A Streamlit web application was developed to demonstrate model deployment. The application loads a trained Logistic Regression model and predicts stroke risk probability, categorizing users into Low, Moderate, or High Risk groups.
Logistic Regression was chosen for deployment due to its slightly better ROC–AUC, clear probability interpretation, and higher interpretability, while AutoML served as a validation framework.

Run the app:
pip install -r requirements.txt
streamlit run app.py

⚠️ This application is intended for educational and screening purposes only.
