# Credit Card Fraud Detection

A comparative machine learning project for detecting fraudulent credit card transactions using Decision Tree, k-Nearest Neighbors (kNN), and Naive Bayes on an imbalanced dataset.

## Project Objective
This project evaluates the effectiveness of three classification models for fraud detection and identifies the most suitable model based on validation and test-set performance.

## Business Problem
Credit card fraud detection is an imbalanced classification problem where missing fraudulent transactions can lead to financial loss, while excessive false alarms may disrupt customer experience and increase review costs.

## Dataset
- Source: Kaggle credit card fraud dataset
- Original training set: approximately 1.3 million records
- Original test set: 555,719 records
- Target variable: `is_fraud`
- Severe class imbalance: fraud rate below 0.6%

## Methodology

### Data preprocessing
- standardized column names
- removed duplicates
- converted target variable to factor
- stratified sampling:
  - 20,000 records for training
  - 50,000 records for test evaluation
- removed irrelevant and leakage-related features

### Feature engineering
- extracted transaction hour, weekday, weekend flag, and daypart
- created behavioural features such as:
  - time since last transaction
  - transaction count within 1 hour and 24 hours
  - amount sum within 1 hour and 24 hours
- computed customer–merchant distance using Haversine distance
- reduced high-cardinality categorical variables
- applied log transformation to transaction amount

### Imbalanced learning
- applied SMOTE / oversampling on the training set
- tuned the fraud ratio to around 5% to balance fraud detection and generalization

### Models compared
- Decision Tree
- k-Nearest Neighbors (kNN)
- Naive Bayes

### Evaluation metrics
- AUC
- Recall
- Precision
- F1-score
- Balanced Accuracy
- Confusion Matrix

## Validation Results

| Model | AUC | Recall | Precision | F1 |
|---|---:|---:|---:|---:|
| Decision Tree | 0.9995 | 0.9851 | 0.9167 | 0.9496 |
| kNN | 0.9960 | 0.9900 | 0.7210 | 0.8344 |
| Naive Bayes | 0.9665 | 0.0000 | NA | NA |

## Final Test Result
The Decision Tree model was selected for final testing on an untouched test set.

- Test AUC: 0.8892
- Recall: 0.7565
- Precision: 0.2955
- F1-score: 0.4250


## Files
- `fraud_detection_models.R`: main R script for preprocessing, feature engineering, modeling, and evaluation
- `project-report.pdf` or `reports/project-report.pdf`: report summary
- `presentation-slides.pdf` or `reports/presentation-slides.pdf`: presentation slides
- `images/`: selected output charts and model visuals

## Tech Stack
- R
- caret
- rpart
- naivebayes
- tidymodels
- recipes
- dplyr
- pROC
- ggplot2

## My Contribution
This was a group academic project. I use this repository to present the project workflow, model comparison logic, and selected outputs as part of my analytics portfolio.

## Notes
The original dataset is not uploaded to this repository due to file size and source restrictions. Please download the dataset separately if you want to reproduce the workflow.
