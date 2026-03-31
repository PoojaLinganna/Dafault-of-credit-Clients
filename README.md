

# Credit Card Default Prediction

## Project Overview

This project predicts the probability of credit card default using demographic and payment history data from the **Default of Credit Card Clients** dataset (Taiwan, 25,000+ records). The goal is to compare machine learning methods for predicting defaults and provide insights for credit risk assessment.

---

## Dataset

* **Source:** UCI / Kaggle – [Default of Credit Card Clients Dataset](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)
* **Size:** 30,000 rows × 25 columns
* **Features:** Demographics (age, gender, education, marital status), credit limit, past 6-month bill amounts, and past 6-month payment amounts.
* **Target:** `default.payment.next.month` (1 = default, 0 = no default)

---

## Objectives

1. Estimate the probability of default for each client.
2. Compare the performance of **Decision Tree**, **Linear SVM**, and **Nonlinear SVM (RBF)** models.
3. Evaluate models using **confusion matrix**, **ROC-AUC**, and **probability estimates**.
4. Provide insights on the factors influencing defaults.

---

## Methodology

### 1. Data Processing

* Imported and inspected the dataset using pandas.
* Explored statistics and distributions of numeric features.
* Handled categorical variables and standardized numeric features.
* Split data into training (80%) and testing (20%) sets using stratified sampling.

### 2. Modeling

* **Decision Tree Classifier:** max depth = 5, min samples per leaf = 100.
* **Linear SVM:** linear kernel, probability estimates enabled.
* **Nonlinear SVM (RBF):** RBF kernel, probability estimates enabled.

### 3. Evaluation

* Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC.
* Visualizations: Confusion matrices and ROC curves.
* Feature importance interpreted from the Decision Tree.

---

## Key Findings

* **Default Rate:** ~22% of clients defaulted.
* **Decision Tree:** Accuracy 0.817, AUC 0.744 – performed well on moderately imbalanced data.
* **Linear SVM:** Accuracy 0.808, AUC 0.685 – lower recall for default cases.
* **Nonlinear SVM (RBF):** Accuracy 0.817, AUC 0.724 – handled nonlinear relationships better.
* **Important Factors:** Payment history, credit limit, and previous bill amounts were strong predictors of default.

---

## Conclusion

Machine learning models can effectively predict credit card defaults and assist in credit risk management. Ensemble or nonlinear models (like Random Forest or RBF SVM) provide slightly better performance in capturing complex patterns in the data. Banks can use these insights for early detection, credit scoring, and tailored financial guidance.

---

## Usage

3. Load the dataset: `df = pd.read_csv('UCI_Credit_Card.csv')`
4. Run preprocessing, modeling, and evaluation scripts to reproduce results.
