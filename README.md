# Smart Recruitment Assistant

## 1. Project Overview

Smart Recruitment Assistant is an AI-powered recruitment screening system. It helps HR teams analyze candidate information and estimate whether a candidate is likely to be looking for a job change. The project combines reproducible preprocessing, supervised machine learning, saved reports, and an interactive Streamlit dashboard.

## 2. Problem Statement

Recruitment teams may need to review many candidate profiles while trying to identify people who are likely to be open to a job change. Automated screening can help prioritize profiles consistently and surface useful patterns for review. The model is a decision-support tool; it does not replace recruiter judgment.

## 3. Project Objectives

- Candidate screening using binary classification.
- Recruitment analytics and candidate distributions.
- Machine-learning prediction and confidence estimation.
- Ranking of the highest-confidence test-set records.
- Explainable model insights through coefficients and feature importance.
- An interactive Streamlit dashboard for HR-oriented review.

## 4. Dataset

The project uses the HR Analytics Job Change dataset files stored in `data/`: `aug_train.csv`, `aug_test.csv`, and `sample_submission.csv`. The repository documentation identifies this as the public HR Analytics Job Change dataset; no external source URL is stored in the project.

The available files contain:

- `aug_train.csv`: 19,158 records and the binary `target` column.
- `aug_test.csv`: 2,129 records without the target column.
- `sample_submission.csv`: submission-format example data.

Main feature categories are candidate and location information, education and experience, company information, and training activity. The original dataset column names are preserved for compatibility.

The target is `target`:

- `0`: Not Looking for Job Change
- `1`: Looking for Job Change

The training target distribution is approximately 75.07% class 0 and 24.93% class 1.

## 5. Data Preprocessing

Preprocessing is implemented by `RecruitmentPreprocessor` in `app/modeling.py` and saved as `models/preprocessing_pipeline.pkl`. The same fitted artifact is loaded by the Streamlit application.

The implemented process is:

1. Convert `experience`: `<1` becomes `0`, `>20` becomes `21`, numeric strings become numeric values, and invalid or missing values become missing numeric values.
2. Impute `city_development_index`, `training_hours`, and converted `experience` with medians fitted on the training data only.
3. Impute categorical features with the literal value `Unknown`.
4. Apply explicit ordinal mappings to `education_level`, `company_size`, and `last_new_job`. Unknown or unmapped values become `-1`.
5. Frequency-encode `city` using frequencies learned from training data only. Unseen cities receive frequency `0`.
6. One-hot encode `gender`, `relevent_experience`, `enrolled_university`, `major_discipline`, and `company_type`. Unknown categories are ignored safely.
7. Scale city frequency, city development index, training hours, and experience with `StandardScaler` fitted on training data only.
8. Use an 80% training and 20% holdout split with `random_state=42` and `stratify=y`.

## 6. Machine Learning Models

- **Logistic Regression:** an interpretable linear classification baseline with directional coefficients.
- **Random Forest:** a decision-tree ensemble for nonlinear relationships with native tree importance.
- **ExtraTrees:** a randomized tree ensemble used as an additional tabular classification candidate.
- **HistGradientBoosting:** a gradient-boosted tree model selected for its overall held-out performance.

Because the target is imbalanced, Logistic Regression, Random Forest, and ExtraTrees use `class_weight="balanced"`. HistGradientBoosting uses balanced training sample weights.

## 7. Model Evaluation

These are the latest measured results from `reports/model_comparison.csv`, calculated on the fixed stratified holdout test split:

| Model | Accuracy | Precision | Recall | F1 | Balanced Accuracy | ROC-AUC | Selection Score |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 75.08% | 50.00% | 78.12% | 60.97% | 76.09% | 80.28% | 73.86% |
| Random Forest | 78.16% | 57.62% | 46.70% | 51.59% | 67.65% | 80.03% | 61.49% |
| ExtraTrees | 78.55% | 55.18% | 74.14% | 63.27% | 77.08% | 80.23% | 73.68% |
| HistGradientBoosting | **79.44%** | 56.44% | 76.65% | **65.01%** | **78.51%** | **82.05%** | **75.55%** |

The selected model is **HistGradientBoosting**. The selection score is the mean of Recall, F1, Balanced Accuracy, and ROC-AUC, so screening quality and class-imbalance behavior are considered alongside Accuracy. Accuracy above 90% has not been achieved legitimately on the current holdout data.

## 8. Dashboard Features

The Streamlit application in `app/app.py` provides five sections:

- **Candidate Prediction:** enter a candidate profile and receive a screening prediction and confidence value.
- **Recruitment Dashboard:** review KPIs, model quality, candidate distributions, and feature importance.
- **Top Candidates:** inspect the top 10 test-set records ranked by selected-model confidence.
- **Model Insights:** review model metrics, Logistic Regression coefficients, and selected-model feature importance.
- **About Project:** read the project purpose, data, preprocessing, models, and limitations.

## 9. Candidate Prediction

Users enter candidate information through categorical selectors and bounded numerical controls. The record is passed through the saved preprocessing artifact and selected model. The application displays the predicted class, probability-based confidence, and a short screening explanation.

## 10. Explainability

- `reports/logistic_coefficients.csv` contains Logistic Regression coefficients and absolute coefficient values. Positive coefficients increase estimated class-1 likelihood; negative coefficients decrease it.
- `reports/feature_importance.csv` contains selected-model importance. HistGradientBoosting uses permutation importance measured with ROC-AUC because it has no native `feature_importances_` attribute.
- `reports/model_comparison.csv` contains the common evaluation table and selection score.

These outputs describe model behavior in this dataset and should not be interpreted as causal conclusions.

## 11. Project Structure

```text
.
├── app/
│   ├── app.py
│   └── modeling.py
├── data/
│   ├── aug_train.csv
│   ├── aug_test.csv
│   └── sample_submission.csv
├── models/
│   ├── extra_trees.pkl
│   ├── hist_gradient_boosting.pkl
│   ├── logistic_regression.pkl
│   ├── preprocessing_pipeline.pkl
│   └── random_forest.pkl
├── notebooks/
│   └── Smart_Recruitment_Assistant_Final.ipynb
├── reports/
│   ├── eda_insights.json
│   ├── feature_importance.csv
│   ├── logistic_coefficients.csv
│   ├── model_comparison.csv
│   └── top_candidates.csv
├── screenshots/
├── .gitignore
├── README.md
├── requirements.txt
└── Smart_Recruitment_Assistant_Sprints_Checklist.md
```

The `screenshots/` directory currently exists but contains no screenshots.

## 12. Installation

From the project root in Windows PowerShell:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
py -3 -m pip install -r requirements.txt
```

## 13. Running the Application

From the project root:

```powershell
py -3 -m streamlit run app\app.py
```

Streamlit normally displays `http://localhost:8501`. This localhost address is for local testing on the current computer and is not a public deployment URL.

## 14. Deployment

The application can later be deployed publicly with Streamlit Community Cloud after repository, data availability, dependency, and security requirements have been reviewed. This project has not been deployed.

## 15. Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Streamlit
- Matplotlib
- Seaborn
- Jupyter

## 16. Limitations

- The current best holdout accuracy is 79.44%, below 90%.
- Results depend on the quality, coverage, and historical nature of the dataset.
- Model confidence is not a guarantee of candidate behavior.
- Predictions should support HR decisions rather than replace human judgment.
- Feature explanations indicate predictive contribution, not causation.
- The Top Candidates report ranks test-set records and does not establish identities beyond the available dataset identifier.

## 17. Future Improvements

- Develop and validate additional meaningful feature engineering.
- Perform broader hyperparameter optimization with training-only cross-validation.
- Evaluate additional tabular models when their dependencies are appropriate.
- Add stronger explainability and subgroup performance analysis.
- Add repeated validation and probability calibration.
- Deploy publicly after operational review.
- Add authentication, authorization, and security controls if sensitive data is handled.

## 18. Screenshots

The `screenshots/` directory exists but is currently empty. No screenshot filenames are referenced because no screenshots are stored in the project.
