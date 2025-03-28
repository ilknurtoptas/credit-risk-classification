# Credit Risk Classification - Machine Learning Model Report

## Overview of the Analysis

The goal of this analysis was to build a machine learning model to predict credit risk and help lenders make more informed decisions about loan approvals. Specifically, the objective was to classify loans as either:

- **0** – Healthy Loan  
- **1** – High-Risk Loan

The dataset included financial information about borrowers, such as:

- Loan size
- Interest rate
- Borrower income
- Debt-to-income ratio
- Number of credit accounts
- Derogatory credit marks
- Total outstanding debt

The dataset includes the following key financial features:

- loan_size
- interest_rate
- borrower_income
- debt_to_income
- num_of_accounts
- derogatory_mark
- total_debt
- loan_status

The target variable was `loan_status`, with value counts showing that most loans were healthy (`0`), indicating class imbalance:

```python
loan_status
0    75036
1     2500
Name: count, dtype: int64

```

### Machine Learning Process:
We followed a standard ML pipeline:
1. Preprocessed the data (cleaning and feature selection)
2. Split the data into training and test sets
3. Trained a **Logistic Regression** model
4. Evaluated performance using classification metrics (accuracy, precision, recall)

Logistic Regression was chosen for its simplicity and interpretability as a baseline model. Additional models may be considered in future iterations to further improve recall on high-risk loans.

---

## Dataset and Model Development Summary

The dataset (77,536 data points) was split into training and testing sets. The training set was used to build a logistic regression model using the `LogisticRegression` module from scikit-learn. This model was then applied to the testing dataset. The purpose of the model was to determine whether a loan to the borrower in the testing set would be low- or high-risk, and the results are summarized below.

This model was trained on an imbalanced dataset with **75,036 low-risk loans** and **2,500 high-risk loans**. Despite the imbalance, the model was able to achieve strong performance metrics, particularly in identifying low-risk loans with high accuracy and precision, while also performing reasonably well in detecting high-risk loans.

---

## Results

* **Machine Learning Model: Logistic Regression**
  * **Accuracy Score**: 0.99
  * **Precision**:
    * Class 0 (healthy loan): 1.00
    * Class 1 (high-risk loan): 0.84
  * **Recall**:
    * Class 0 (healthy loan): 0.99
    * Class 1 (high-risk loan): 0.94
  * **F1 Score**:
    * Class 0: 1.00
    * Class 1: 0.89

---

## Summary

The logistic regression model performed exceptionally well in identifying **healthy loans** and quite well in predicting **high-risk loans**. With an overall **accuracy of 99%**, it is a highly reliable model. However, there is a slight drop in performance for high-risk loans, particularly in terms of precision (0.84), which suggests a few false positives (predicting high-risk loans incorrectly).

### Recommendation:
We recommend using the **logistic regression model** for credit risk prediction due to its high performance across metrics and its simplicity. That said, if the company is more concerned with accurately identifying high-risk loans (label 1), it might be worth exploring more complex models (e.g., Random Forest, XGBoost) or techniques like SMOTE to handle class imbalance.
