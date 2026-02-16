# 🔍 Explainable Credit Risk Modeling – German Credit Dataset

<p align="center">
  <img src="https://img.shields.io/badge/Models-LogReg%20|%20Decision%20Tree%20|%20RF%20|%20XGBoost-blue" />
  <img src="https://img.shields.io/badge/Focus-Credit%20Risk%20Prediction-red" />
  <img src="https://img.shields.io/badge/Evaluation-ROC--AUC%20%26%20PR--AUC-green" />
  <img src="https://img.shields.io/badge/Explainability-SHAP%20%26%20LIME-purple" />
</p>


## Project Overview

This project implements an end-to-end **credit risk analysis pipeline** using the **Statlog (German Credit) dataset**, with a strong emphasis on **Explainable Artificial Intelligence (XAI)**. The primary objective is to build reliable predictive models for credit risk assessment while ensuring transparency and interpretability of model decisions, which is critical in financial applications.

The workflow spans exploratory data analysis, model development and evaluation, cross-validation-based comparison, and global and local explainability using model-agnostic interpretation techniques.

---

## Dataset

- **Name:** Statlog (German Credit Data)  
- **Source:** UCI Machine Learning Repository  
- **Dataset URL:** [URL](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data)
- **Number of samples:** 1,000  
- **Number of features:** 20 (categorical and numerical)  
- **Target variable:** Credit risk (good vs bad credit)

The dataset represents a realistic financial decision-making scenario and includes attributes related to credit history, loan characteristics, and demographic information.

---


## Dashboard Demo

The project includes an interactive dashboard that visualizes:
- Credit risk predictions  
- Feature-level contributions  
- Global and local explainability outputs  

Below is a demonstration of the dashboard functionality:

![Dashboard Demo](./dashboard/Demo.gif)

---

## Methodology

### 1. Exploratory Data Analysis
- Data quality checks and preprocessing  
- Feature inspection and distribution analysis  
- Class balance assessment  
- Correlation and relationship analysis  

### 2. Model Development
The following machine learning models are trained and evaluated:
- Logistic Regression  
- Decision Tree  
- Random Forest  
- XGBoost  

Model performance is assessed using ROC-AUC and Precision–Recall AUC metrics, with cross-validation to evaluate generalization performance.

### 3. Cross-Validation Results

The table below summarizes the mean and standard deviation of ROC-AUC scores across cross-validation folds:

| Model                | CV ROC-AUC Mean | CV ROC-AUC Std |
|---------------------|----------------|----------------|
| Random Forest       | 0.7858         | 0.0610         |
| Logistic Regression | 0.7811         | 0.0699         |
| XGBoost             | 0.7538         | 0.0529         |
| Decision Tree       | 0.5839         | 0.0417         |

### 4. Explainable AI (XAI)

To ensure transparency and interpretability of model predictions, the following explainability techniques are applied:
- **Global explanations** using SHAP to identify overall feature importance  
- **Local explanations** using SHAP and LIME to explain individual predictions  

---
## Conclusion

- Given the relatively small and well-structured nature of the dataset, standard machine learning models are able to capture most underlying patterns without extensive hyperparameter tuning. 
- Among the evaluated models, **Random Forest** demonstrates the strongest overall performance, achieving the highest ROC-AUC and Precision–Recall AUC scores. Cross-validation results further confirm its robustness, with consistent performance and low variance across folds.
- The **Decision Tree** model performs poorly due to high variance and overfitting, resulting in unstable predictions on unseen data.
- **XGBoost** performs competitively but does not outperform Random Forest in this setting; on a categorical-heavy dataset without extensive tuning, it tends to produce sharper and less stable probability estimates.

Based on these findings, **Random Forest is selected as the final model** for subsequent explainability analysis.

---

## References

- [SHAP: SHapley Additive Explanations](https://shap.readthedocs.io/)
- [LIME: Local Interpretable Model-Agnostic Explanations](https://github.com/marcotcr/lime)


