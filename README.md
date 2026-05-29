# Uptrail-Week-3
Churn prediction analysis using logistic regression and linear regression in Python — class imbalance handling, statistical testing and model evaluation on subscriber data

**Project Overview**

Analysis of subscriber churn for StreamWorks Media, a fictional streaming platform. The goal was to identify churn patterns, build a classification model to predict at-risk customers, and investigate what drives watch time using linear regression.
Dataset: 1,499 subscriber records with demographic, behavioural and subscription-level attributes.

**Files**

streamworks_user_data.csv — raw subscriber dataset

StreamWorks_Churn_Analysis.ipynb — full Python notebook covering cleaning, feature engineering, statistical tests and modelling

StreamWorks_Churn_Report.docx — business insights report with full findings and recommendations


**Data Cleaning**
The dataset was largely well-formed on arrival. The main issues addressed were missing values across several columns:

Categorical nulls filled with mode to preserve existing distribution
Numerical nulls filled with median — preferred over mean for behavioural metrics susceptible to outliers
Date parsing errors handled with errors='coerce' and median imputation on affected tenure values
No duplicate rows identified


**Feature Engineering**
*Two features were derived to support lifecycle-stage analysis:*

*tenure_days* — days between signup and last active date
*is_loyal* — binary flag for customers active more than 180 days

*Categorical encoding prior to modelling:*

Ordinal encoding for subscription_type (Basic → Standard → Premium = 0, 1, 2) preserving natural tier hierarchy
One-hot encoding for country and gender where no ordinal relationship exists


*Statistical Testing*

Chi-square tests on categorical variables and t-tests on continuous variables were run before modelling to identify meaningful associations with churn. No feature returned a statistically significant result, with gender producing the lowest p-value (p = 0.05) — marginal at best.

*Model Results* 

Logistic Regression — Unbalanced Baseline

73% accuracy but predicted every customer as not churned
Entirely useless for the actual business problem — the model learned to ignore churners altogether

*Logistic Regression* — class_weight='balanced'

Accuracy fell to 53% but churn recall improved from 0% to 48%
Correctly identified 39 of 81 actual churners
Recall was prioritised over accuracy — a missed churner is a lost subscriber with no opportunity for intervention

*Random Forest*

Tested as an alternative but exhibited the same majority-class bias and was discarded

*Linear Regression* — Watch Hours

R² of 0.016, RMSE of 23.2 hours
Predictions flat-lined around the mean regardless of actual values
Key drivers such as content preferences and viewing history are absent from the dataset


**Key Findings**

No feature in the available dataset showed a statistically significant association with churn
Country-level features dominated the logistic regression coefficients, likely reflecting uneven country representation rather than a genuine geographic effect
Promotions showed no significant relationship with churn — current strategy should be reviewed
The dataset is insufficient to support reliable churn prediction without enrichment


**Recommendations**

Enrich the dataset with content preferences, viewing history and session frequency data
Explore SMOTE oversampling alongside class weighting to better address the class imbalance
Review promotion targeting — no evidence it is reducing cancellations
Treat any model-driven decisions with caution until richer features are available


**Tools Used**

Python — pandas, scikit-learn, seaborn, matplotlib, scipy
Jupyter Notebook — primary analysis environment


*The original notebook was lost due to a system failure. It was reconstructed with the assistance of AI tooling based on the original methodology and report, and validated against documented findings.*
