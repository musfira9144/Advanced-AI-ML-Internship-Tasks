# Task 2: End-to-End Machine Learning Pipeline for Customer Churn Prediction

## 📌 Objective
The objective of this task is to build a **reusable and production-ready machine learning pipeline**
to predict customer churn using the **Telco Customer Churn Dataset**.  
The pipeline automates data preprocessing, model training, hyperparameter tuning, evaluation,
and model export using Scikit-learn.

---

## 📊 Dataset
**Telco Customer Churn Dataset**

- Customer demographic, service usage, and billing data
- Target variable: **Churn** (`Yes` / `No`)
- Total features: **21**

---

## 🛠️ Methodology / Approach

### 1. Data Preprocessing
- Handled missing values in numerical features (`TotalCharges`)
- Used **ColumnTransformer** for preprocessing:
  - Numerical features → Standard Scaling
  - Categorical features → One-Hot Encoding
- Ensured preprocessing is part of the pipeline for consistency and reusability

### 2. Machine Learning Pipeline
- Built an end-to-end pipeline using **Scikit-learn Pipeline API**
- Models implemented:
  - Logistic Regression
  - Random Forest Classifier

### 3. Hyperparameter Tuning
- Applied **GridSearchCV** with 5-fold cross-validation
- Tuned Random Forest parameters:
  - `n_estimators`
  - `max_depth`
  - `min_samples_split`

### 4. Model Evaluation
- Evaluation metrics used:
  - Accuracy
  - Precision
  - Recall
  - F1-score
- Model performance visualized using a **confusion matrix**

### 5. Model Export
- Final trained pipeline exported using **joblib**
- Saved as:churn_prediction_pipeline.joblib


- Exported pipeline can be reused directly for predictions on new data

---

## 📈 Key Results / Observations
- Achieved approximately **80% accuracy** on the test dataset
- Model performs well for non-churn customers
- Lower recall for churn class indicates class imbalance
- Pipeline-based approach ensures:
- Reproducibility
- Clean deployment
- Easy future integration

---


