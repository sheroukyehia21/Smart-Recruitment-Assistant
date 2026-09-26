# 🤖 Smart Recruitment Assistant

## 📌 Overview

🚀 **Live Demo:** [Open the Streamlit App](https://smart-recruitment-assistant-hqqtcwyajd86f2bgf6lbhn.streamlit.app/)
📊 **Dataset:** [HR Analytics Job Change of Data Scientists – Kaggle](https://www.kaggle.com/datasets/arashnic/hr-analytics-job-change-of-data-scientists)

Smart Recruitment Assistant is an AI-powered candidate screening tool built for HR teams. It takes candidate information (education, experience, company background, training activity, etc.) and predicts how likely a candidate is to be looking for a job change, using a supervised machine-learning classifier trained on the HR Analytics Job Change dataset.

The project is meant as a decision-support tool for recruiters — it surfaces a prediction and a confidence score to help prioritize review, not to replace recruiter judgment. The core AI component is a binary classification pipeline (four candidate models were trained and compared), wrapped in an interactive Streamlit dashboard.


## 🎯 Objectives

- Screen candidates using binary classification (likely / not likely to look for a job change).
- Provide recruitment analytics and candidate distribution insights.
- Generate a prediction with a confidence score for each candidate.
- Rank the highest-confidence records in the test set.
- Offer explainable model insights via coefficients and feature importance.
- Present all of the above through an interactive, HR-oriented Streamlit dashboard.

## ✨ Key Features

- Candidate screening prediction from a manually entered profile, with a probability-based confidence score.
- A recruitment dashboard with KPIs, model-quality metrics, candidate distributions, and feature importance.
- A "Top Candidates" view ranking the 10 highest-confidence records from the test set.
- A model-insights view showing Logistic Regression coefficients and the selected model's feature importance.
- An "About" page documenting the project's purpose, data, preprocessing, models, and limitations.
- A reproducible preprocessing pipeline saved as a single artifact and reused identically between training and the app.

## 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Streamlit
- Matplotlib
- Seaborn
- Jupyter

(As listed in `requirements.txt`: `pandas==2.3.3`, `numpy==2.0.2`, `scikit-learn==1.6.1`, `matplotlib>=3.7`, `seaborn>=0.12`, `joblib==1.5.3`, `streamlit==1.50.0`, `jupyter>=1.0`. `runtime.txt` pins Python `3.11.9`.)

## 📂 Project Structure

```
.
├── Notebooks/
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
├── reports/
│   ├── eda_insights.json
│   ├── feature_importance.csv
│   ├── logistic_coefficients.csv
│   ├── model_comparison.csv
│   └── top_candidates.csv
├── screenshots/               # currently empty
├── hr analytics.zip
├── requirements.txt
├── runtime.txt
├── Smart_Recruitment_Assistant_Sprints_Checklist.md
└── Smart_Recruitment_Assistant_Technical_Documentation.docx
```

> Note: the top level of the repository is a `Notebooks/` folder. The project's own internal documentation additionally states its training notebook is `Notebooks/Smart_Recruitment_Assistant_Final.ipynb`; this notebook path could not be independently browsed while writing this README (GitHub blocks automated folder browsing), so treat the exact notebook filename as documented-but-not-independently-verified.

## 📊 Dataset

The project uses the HR Analytics Job Change dataset, stored in `data/` as `aug_train.csv`, `aug_test.csv`, and `sample_submission.csv`. This is documented as the public HR Analytics Job Change dataset; no external source URL is stored in the project itself.

- `aug_train.csv`: 19,158 records, including the binary `target` column.
- `aug_test.csv`: 2,129 records, without the target column.
- `sample_submission.csv`: example submission-format data.

Main feature categories: candidate and location information, education and experience, company information, and training activity. Original dataset column names are preserved.

Target column: `target`
- `0` — Not looking for a job change
- `1` — Looking for a job change

Training set class balance: **≈75.07% class 0 / ≈24.93% class 1** (imbalanced).

## 🔄 Data Preprocessing

Implemented by `RecruitmentPreprocessor` in `app/modeling.py`, and saved as a single fitted artifact at `models/preprocessing_pipeline.pkl`, which the Streamlit app also loads at inference time.

1. Convert `experience`: `<1` → `0`, `>20` → `21`, numeric strings → numbers, invalid/missing → missing.
2. Impute `city_development_index`, `training_hours`, and converted `experience` with medians fitted on the training data only.
3. Impute categorical features with the literal value `"Unknown"`.
4. Apply explicit ordinal mappings to `education_level`, `company_size`, and `last_new_job`; unmapped/unknown values become `-1`.
5. Frequency-encode `city` using frequencies learned from training data only; unseen cities get frequency `0`.
6. One-hot encode `gender`, `relevent_experience`, `enrolled_university`, `major_discipline`, and `company_type`, ignoring unseen categories safely.
7. Scale city frequency, city development index, training hours, and experience with `StandardScaler` fitted on training data only.
8. Split 80% train / 20% holdout with `random_state=42` and `stratify=y`.

## 🔍 Exploratory Data Analysis

EDA outputs are saved to `reports/eda_insights.json`. The documented findings center on the target's class imbalance (~75/25) and how candidate education, experience, and training activity relate to the target — these are summarized qualitatively in the project's own documentation rather than as a separate narrative report file.

## 🤖 Machine Learning Models

Four models were trained and compared on the same preprocessed data/split:

- **Logistic Regression** — interpretable linear baseline; produces directional coefficients for explainability. Uses `class_weight="balanced"` to address class imbalance.
- **Random Forest** — tree ensemble for nonlinear relationships, with native feature importance. Uses `class_weight="balanced"`.
- **ExtraTrees** — randomized tree ensemble used as an additional tabular candidate. Uses `class_weight="balanced"`.
- **HistGradientBoosting** — gradient-boosted tree model; uses balanced sample weights during training instead of `class_weight`. Selected as the final model (see below).

## 📈 Model Evaluation & Comparison

Results below are from `reports/model_comparison.csv`, measured on the fixed, stratified holdout split:

| Model | Accuracy | Precision | Recall | F1 Score | Balanced Accuracy | ROC-AUC | Selection Score |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 75.08% | 50.00% | 78.12% | 60.97% | 76.09% | 80.28% | 73.86% |
| Random Forest | 78.16% | 57.62% | 46.70% | 51.59% | 67.65% | 80.03% | 61.49% |
| ExtraTrees | 78.55% | 55.18% | 74.14% | 63.27% | 77.08% | 80.23% | 73.68% |
| **HistGradientBoosting** | **79.44%** | 56.44% | 76.65% | **65.01%** | **78.51%** | **82.05%** | **75.55%** |

**HistGradientBoosting** was selected. The "Selection Score" is the mean of Recall, F1, Balanced Accuracy, and ROC-AUC — chosen deliberately over raw Accuracy so that screening quality and class-imbalance behavior are weighted alongside overall correctness. No model reached 90%+ accuracy on this holdout data.

## 🎯 Candidate Prediction

The user enters a candidate profile through categorical selectors and bounded numeric inputs in the Streamlit app. That input is passed through the saved preprocessing pipeline (identical transformations to training) and then through the selected model (HistGradientBoosting). The app returns the predicted class, a probability-based confidence value, and a short screening explanation.

## 📊 Streamlit Dashboard

The app (`app/app.py`) has five sections:

- **Candidate Prediction** — enter a candidate profile, get a prediction and confidence value.
- **Recruitment Dashboard** — KPIs, model quality, candidate distributions, feature importance.
- **Top Candidates** — the top 10 test-set records ranked by the selected model's confidence.
- **Model Insights** — model metrics, Logistic Regression coefficients, selected-model feature importance.
- **About Project** — purpose, data, preprocessing, models, and limitations.

## 💾 Saved Models / Artifacts

- `models/logistic_regression.pkl`
- `models/random_forest.pkl`
- `models/extra_trees.pkl`
- `models/hist_gradient_boosting.pkl` (selected model)
- `models/preprocessing_pipeline.pkl` (fitted preprocessing artifact used by both training and the app)
- `reports/logistic_coefficients.csv` — Logistic Regression coefficients and absolute coefficient values (sign indicates direction of effect on class-1 likelihood).
- `reports/feature_importance.csv` — selected-model (HistGradientBoosting) importance, computed via permutation importance on ROC-AUC, since this model has no native `feature_importances_`.
- `reports/model_comparison.csv` — the evaluation table above.
- `reports/top_candidates.csv` — the ranked top test-set records.
- `reports/eda_insights.json` — saved EDA findings.

## 🚀 How to Run

```bash
# 1. Clone the repository
git clone https://github.com/sheroukyehia21/Smart-Recruitment-Assistant.git
cd Smart-Recruitment-Assistant

# 2. Create and activate a virtual environment (Windows PowerShell)
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Install dependencies
py -3 -m pip install -r requirements.txt

# 4. Run the Streamlit app
py -3 -m streamlit run app\app.py
```

Streamlit will normally open at `http://localhost:8501`. This is a local address for testing on your own machine — the project has not been deployed publicly.

## 📋 Example Workflow

1. **Candidate data** — a profile is entered via the Streamlit form (or comes from the test set for the Top Candidates view).
2. **Preprocessing** — the saved `preprocessing_pipeline.pkl` applies the exact training-time transformations (imputation, ordinal mapping, frequency encoding, one-hot encoding, scaling).
3. **Model** — the preprocessed record is passed to the selected HistGradientBoosting model.
4. **Prediction** — the model outputs a class and probability.
5. **Output** — the app displays the predicted class, a confidence percentage, and a short explanation.

## 📌 Project Limitations

- Best holdout accuracy is 79.44%, below 90%.
- Results depend on the coverage and historical nature of the underlying dataset.
- A confidence score is not a guarantee of a candidate's actual behavior.
- Predictions are meant to support HR decisions, not replace human judgment.
- Feature importance/coefficients show predictive contribution, not causation.
- The Top Candidates ranking reflects test-set records and does not establish identities beyond the dataset's own identifier.

## 🔮 Future Improvements

- Additional feature engineering, validated properly.
- Broader hyperparameter optimization using training-only cross-validation.
- Evaluating further tabular models where dependencies allow.
- Stronger explainability and subgroup performance analysis.
- Repeated validation and probability calibration.
- Public deployment (e.g., Streamlit Community Cloud) after an operational review.
- Authentication, authorization, and security controls if the app is extended to handle sensitive data.

## 👩‍💻 My Contribution

This was a collaborative project. The repository's own sprint-planning document assigns work by generic role placeholders (data cleaning/preprocessing, EDA, individual models, and dashboard/integration) rather than by name, and no other file in the repository documents individual authorship. Individual responsibilities should be specified by the project team.

## 📄 Project Deliverables

- Trained and compared candidate-screening models (Logistic Regression, Random Forest, ExtraTrees, HistGradientBoosting), with the best model selected.
- Saved model artifacts and preprocessing pipeline (`models/`).
- Evaluation, coefficients, feature importance, EDA, and ranking reports (`reports/`).
- Interactive Streamlit recruitment screening application (`app/`).
- Training/analysis notebook(s) (`Notebooks/`).
- Technical documentation (`Smart_Recruitment_Assistant_Technical_Documentation.docx`) and sprint checklist (`Smart_Recruitment_Assistant_Sprints_Checklist.md`).
- Raw dataset archive (`hr analytics.zip`).
No license file was found in the repository, so no license is stated here. Add one (e.g., MIT) if you intend this project to be reused by others.
