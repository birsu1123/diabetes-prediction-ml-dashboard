# Diabetes Prediction with Tree-Based Machine Learning

An interactive **Python Shiny dashboard** for exploring a diabetes dataset, tuning tree-based classifiers, and comparing their predictive performance. The project evaluates **Decision Tree**, **Random Forest**, and **Gradient Boosting** models using patient health measurements.

## Project overview

The application combines exploratory data analysis with a comparative machine learning workflow. It uses a publicly available dataset of **390 observations** and predicts a binary diabetes outcome from demographic, anthropometric, biochemical, and cardiovascular measurements.

The dashboard has three sections:

- **Data:** Preview the dataset and explore feature relationships, distributions, box plots, and correlations.
- **Hyperparameter Tuning:** Inspect 10-fold cross-validation results for decision-tree pruning (`ccp_alpha`), random-forest feature selection (`max_features`), and gradient-boosting learning rate.
- **Comparative Analysis:** Compare classification metrics, confusion matrices, ROC curves, tree visualisations, and feature importance across the three models.

## Dataset

**Source:** [Predict Diabetes Based on Diagnostic Measures — Kaggle](https://www.kaggle.com/datasets/houcembenmansour/predict-diabetes-based-on-diagnostic-measures), published by Houcem Benmansour.

The target is `diabetes` (`0` = no diabetes, `1` = diabetes). The application removes the patient identifier and uses the remaining health measurements as predictors, including age, gender, glucose, cholesterol, BMI, waist-to-hip ratio, and blood pressure.

The dataset is downloaded through `kagglehub` when the application starts.

## Methodology

The project follows a supervised machine learning workflow to predict diabetes status from patient health measurements. The data are cleaned and prepared for modelling, then divided into training and test sets. Class imbalance is addressed during training, and three tree-based classification algorithms are developed and compared.

Hyperparameter tuning uses cross-validation to assess model settings. Model performance is evaluated on the test set using classification metrics and visualisations, including confusion matrices and ROC curves.

## Results

The accompanying project report presents the following test-set results:

| Model | Accuracy | ROC-AUC |
|---|---:|---:|
| Decision Tree | 0.87 | 0.84 |
| Random Forest | 0.89 | 0.88 |
| Gradient Boosting | 0.87 | 0.90 |

Random Forest has the highest reported test accuracy, while Gradient Boosting has the highest reported ROC-AUC. Performance for the diabetes class is weaker than for the no-diabetes class across all three models. These results are based on a small test set and should not be interpreted as evidence of clinical suitability.

## Technology stack

- **Python**
- **Shiny for Python** — interactive web application
- **scikit-learn** — preprocessing, model fitting, cross-validation, and evaluation
- **imbalanced-learn** — SMOTENC oversampling
- **pandas** — data processing
- **Matplotlib and Seaborn** — visualisations
- **KaggleHub** — dataset retrieval

## Project files

```text
.
├── app.py       # Shiny dashboard, data processing, and machine learning models
├── Report.pdf   # Accompanying analysis and discussion
└── README.md    # Project documentation
```

## Data attribution

Benmansour, H. (2021). *Predict diabetes based on diagnostic measures*. Kaggle. https://www.kaggle.com/datasets/houcembenmansour/predict-diabetes-based-on-diagnostic-measures
