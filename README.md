# ML Assignment 2 – Bank Marketing: Term-Deposit Classification

## a. Problem statement

A bank runs direct-marketing telephone campaigns to sell term deposits, and calling every client is expensive. The goal of this project is to build a binary classifier that predicts, from a client's profile and campaign history, whether the client will subscribe to a term deposit (y = yes / no). Five classification models are trained on the same preprocessing and evaluated on the same held-out test set using six metrics, so that their performance can be compared fairly.

## b. Dataset description

The dataset used is `ml-assignment-2/bank.csv`, the UCI Bank Marketing dataset, stored as semicolon-separated values. It has 4,521 instances, 16 input features, and 1 target column (y), with no missing values. The target, y, indicates whether the client subscribed to a term deposit, and the classes are imbalanced: 4,000 clients said no (88.48%) and 521 said yes (11.52%). This is encoded as no = 0 and yes = 1.

The 7 numeric features are age, balance, day, duration, campaign, pdays, and previous. The 9 categorical features are job, marital, education, default, housing, loan, contact, month, and poutcome.

The data was split in a stratified 80/20 manner with random_state=42, giving 3,616 training rows and 905 test rows, with the positive rate holding steady at 11.53% in training and 11.49% in test. The test portion, including the target column, is saved separately as `ml-assignment-2/test_data.csv` so it can be uploaded into the Streamlit app.

For preprocessing, a ColumnTransformer applies StandardScaler to the numeric features and OneHotEncoder(handle_unknown='ignore') to the categorical features. This transformer is wrapped together with each classifier in its own Pipeline, and it is fit on the training split only.

## c. Github Repository Link

GitHub repository: https://github.com/2026ak05179-jebin/ml_assignment_2

## d. Model comparison (905-row test set)

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8928 | **0.8913** | 0.5636 | 0.2981 | 0.3899 | 0.3579 |
| Decision Tree | 0.8586 | 0.6942 | 0.4032 | **0.4808** | **0.4386** | **0.3602** |
| KNN | 0.8840 | 0.7692 | 0.4872 | 0.1827 | 0.2657 | 0.2477 |
| Naive Bayes | 0.8210 | 0.7881 | 0.3092 | 0.4519 | 0.3672 | 0.2737 |
| SVM | **0.8939** | 0.8830 | **0.5909** | 0.2500 | 0.3514 | 0.3374 |

Precision, recall and F1 are for the positive class (yes). Best value per column in bold.

## e. Observations

Always predicting no would already score 88.48% accuracy, so accuracy alone does not say much here. AUC, MCC and F1 give a clearer picture of how each model is actually performing.

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best AUC (0.8913), and second-best on accuracy, F1, and MCC. Precise (0.56) but conservative — finds only about 30% of subscribers. Fast to train, small model file, most balanced performer overall. |
| Decision Tree | Best recall (0.48), best F1 (0.44), and, by a narrow margin, best MCC (0.3602 against 0.3579). But worst AUC (0.69) and lower accuracy (0.859) — its probability scores rank clients poorly, likely overfitting. |
| KNN | Weakest model overall. Lowest recall (0.18), F1 (0.27), and MCC (0.25). Accuracy of 0.884 is no better than predicting the majority class, so in practice it mostly predicts no. |
| Naive Bayes | Lowest accuracy (0.821) and lowest precision (0.31), with fairly high recall (0.45). Flags many clients as likely subscribers but is often wrong. The independence assumption does not fit well with the one-hot encoded, correlated features here. AUC of 0.79 sits mid-table. |
| Support Vector Machine | Best accuracy (0.8939) and best precision (0.59), with AUC close to Logistic Regression (0.883). Recall is low (0.25), so it misses most actual subscribers. Also the slowest to train and predict with. |
| Overall, Winner for your dataset? | Logistic Regression. Best AUC, within 0.0011 of the top accuracy, and within 0.0023 of the top MCC — while being the simplest and cheapest model to run. Decision Tree edges it out on MCC, F1, and recall, but the MCC margin is negligible and its AUC is far behind. If catching more subscribers matters more than precision, Decision Tree or Naive Bayes would be the better pick. |

One caveat worth noting: these results come from a single split with only about 104 positive test cases, so small differences, such as Logistic Regression against SVM, or Logistic Regression against Decision Tree on MCC, should not be treated as statistically solid.