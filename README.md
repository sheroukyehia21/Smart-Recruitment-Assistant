# 🤖 Smart Recruitment Assistant — AI Screening System

## 📌 Project Overview

**Smart Recruitment Assistant** is an AI-powered recruitment screening system designed to help HR teams evaluate job candidates and decide whether a candidate should move to the next stage of the hiring process.

The system uses candidate information such as **education, experience, training, employment history, and other relevant features** to predict the candidate's suitability.

The project also provides:

* 📊 Exploratory Data Analysis (EDA)
* 🤖 Multiple Machine Learning Models
* 📈 Model Performance Comparison
* 🔍 Feature Importance & Insights
* 🎯 Candidate Prediction with Confidence Score
* 🏆 Optional Top-10 Candidate Ranking
* 📊 Interactive Streamlit Dashboard

---

## 🎯 Project Objectives

The main objectives of this project are to:

1. Clean and preprocess recruitment data.
2. Explore candidate characteristics and identify useful patterns.
3. Build Machine Learning models for candidate screening.
4. Compare different models using standard evaluation metrics.
5. Identify the most important factors affecting recruitment predictions.
6. Provide an easy-to-use interface for HR users.
7. Help HR teams make faster and more data-driven screening decisions.

---

## 🛠️ Tech Stack

The project was developed using:

* **Python**
* **Pandas** — Data manipulation and analysis
* **NumPy** — Numerical operations
* **Scikit-learn** — Machine Learning and preprocessing
* **Matplotlib** — Data visualization
* **Seaborn** — Statistical visualization
* **Joblib / Pickle** — Model saving and loading
* **Jupyter Notebook / Google Colab** — Development and experimentation
* **Streamlit** — Interactive web application
* **Git & GitHub** — Version control and team collaboration

---

## 📂 Project Structure

```text
smart-recruitment-assistant/
│
├── data/
│   └── aug_train.csv
│
├── Smart_Recruitment_Assistant.ipynb
│
├── models/
│   ├── logistic_regression.pkl
│   └── random_forest.pkl
│
├── app/
│   └── app.py
│
├── reports/
│   └── performance_comparison.md
│
├── requirements.txt
│
└── README.md
```

---

## 📓 Notebook Structure

The main notebook is divided into several sections:

### Section 0 — Setup

Importing the required Python libraries and preparing the project environment.

### Section 1 — Data Understanding

Initial exploration of the dataset using:

* `info()`
* `describe()`
* `isnull().sum()`
* Target variable analysis

### Section 2 — Data Cleaning & Preprocessing

The preprocessing stage includes:

* Handling missing values
* Encoding categorical features
* Feature engineering
* Feature scaling
* Train/Test splitting
* Handling class imbalance when necessary

The final output of this section is:

```text
X_train
X_test
y_train
y_test
```

These datasets are used by the Machine Learning models.

### Section 3 — Exploratory Data Analysis

EDA is used to understand the candidate data through:

* Target distribution
* Feature vs. target relationships
* Correlation analysis
* Candidate education distribution
* Experience distribution
* Training-hours distribution
* Visualizations and statistical insights

---

## 🤖 Machine Learning Models

### 1. Logistic Regression

Logistic Regression is used as one of the baseline classification models.

The model is evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

The trained model is saved as:

```text
models/logistic_regression.pkl
```

---

### 2. Random Forest

Random Forest is used as a second classification model to provide a more flexible approach to candidate screening.

The model is evaluated using the same metrics:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

Feature importance is also extracted to identify the factors that contribute most to the model's decisions.

The trained model is saved as:

```text
models/random_forest.pkl
```

---

## 📊 Model Comparison

The Logistic Regression and Random Forest models are compared using the same test dataset.

The comparison includes:

| Metric    | Logistic Regression | Random Forest |
| --------- | ------------------: | ------------: |
| Accuracy  |                   — |             — |
| Precision |                   — |             — |
| Recall    |                   — |             — |
| F1 Score  |                   — |             — |

The final model used by the application is selected based on the overall evaluation results, with particular attention to the metrics that are most important for recruitment screening.

---

## 🔍 Candidate Screening

The application allows an HR user to enter candidate information through a simple interface.

The system then:

```text
Candidate Information
        ↓
Preprocessing
        ↓
Trained ML Model
        ↓
Prediction
        ↓
Confidence Score
```

The prediction produces one of two outcomes:

### 🟢 Recommended

The candidate is predicted to be suitable for moving to the next recruitment stage.

### 🔴 Not Recommended

The candidate is predicted not to meet the model's screening criteria for the next stage.

The application also displays a confidence score for the prediction.

---

## 🏆 Top-10 Candidate Ranking

As an optional feature, the system can rank candidates according to their prediction probability.

The system uses:

```python
predict_proba()
```

to obtain prediction probabilities and can then display the **Top 10 candidates** with the highest predicted suitability.

---

## 📊 Dashboard

The Streamlit dashboard provides HR users with useful insights such as:

* Candidate statistics
* Target distribution
* Important candidate features
* Model performance
* Feature importance
* Recruitment insights
* Optional Top-10 candidate ranking

---

## 🚀 Running the Project

### 1. Clone the Repository

```bash
git clone <REPOSITORY_URL>
cd smart-recruitment-assistant
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit Application

```bash
streamlit run app/app.py
```

The application will open in your browser.

---

## 📦 Requirements

The main dependencies include:

```text
pandas
numpy
scikit-learn
matplotlib
seaborn
joblib
streamlit
```

---




## ⚠️ Important Note

The Streamlit application must apply **exactly the same preprocessing steps** used during model training.

This includes:

* Categorical encoding
* Feature engineering
* Feature scaling
* Feature order

Using different preprocessing during prediction may lead to incorrect model results.

---

## 🔮 Future Improvements

Possible future improvements include:

* Deploying the application online
* Adding more Machine Learning models
* Improving candidate ranking
* Adding explainable AI features
* Adding authentication for HR users
* Connecting the system to a recruitment database
* Adding CV/Resume parsing
* Integrating Natural Language Processing (NLP)
* Adding automated candidate recommendations

---

## 📄 Project Deliverables

The final project includes:

* ✅ Candidate Screening Models
* ✅ Logistic Regression Model
* ✅ Random Forest Model
* ✅ Model Performance Comparison
* ✅ Recruitment Dashboard
* ✅ Candidate Prediction Interface
* ✅ Confidence Score
* ✅ Feature Importance Insights
* ✅ Saved ML Models
* ✅ Streamlit Application
* ✅ Project Documentation
* ⭐ Optional Top-10 Candidate Ranking
* ⭐ Optional Streamlit Deployment

---

## 👩‍💻 Project Status

**Status:** In Development

This project was developed as a collaborative Machine Learning and Data Science project focusing on automated recruitment screening and data-driven candidate evaluation.
