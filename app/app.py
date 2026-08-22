from __future__ import annotations

import json
from pathlib import Path
import sys

import joblib
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELING_PATH = PROJECT_ROOT / "modeling.py"
if not MODELING_PATH.exists():
    MODELING_PATH = Path(__file__).resolve().with_name("modeling.py")
if not MODELING_PATH.exists():
    raise ImportError(f"Could not find modeling.py at {PROJECT_ROOT} or {Path(__file__).resolve().parent}")
sys.path.insert(0, str(MODELING_PATH.parent))

from modeling import (
    CITY_FEATURE,
    ID_COLUMN,
    NOMINAL_FEATURES,
    ORDINAL_FEATURES,
    ORDINAL_MAPPINGS,
    RecruitmentPreprocessor,
    project_root,
)


ROOT = project_root()
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
TARGET_LABELS = {0: "Not Looking for Job Change", 1: "Looking for Job Change"}


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@700;800&display=swap');
        :root { --ink: #17232d; --muted: #64727d; --teal: #087f8c; --teal-dark: #075c68; --mint: #e7f4f1; --line: #dce6e8; --paper: #f6f9f8; --coral: #e47b61; }
        .stApp { background: var(--paper); color: var(--ink); font-family: 'DM Sans', sans-serif; }
        [data-testid="stHeader"] { background: transparent; }
        [data-testid="stSidebar"] { background: #153842; border-right: 0; }
        [data-testid="stSidebar"] * { color: #eef8f6; }
        [data-testid="stSidebar"] .stRadio label { padding: .55rem .7rem; border-radius: 9px; }
        [data-testid="stSidebar"] .stRadio label:hover { background: rgba(255,255,255,.1); }
        h1, h2, h3 { font-family: 'Manrope', sans-serif; color: var(--ink); letter-spacing: 0; }
        h1 { font-size: 2.35rem; margin-bottom: .15rem; }
        h2 { margin-top: 1.2rem; }
        .hero { padding: 1.25rem 1.5rem; border: 1px solid var(--line); border-radius: 16px; background: linear-gradient(115deg, #ffffff 0%, #e8f5f2 100%); margin-bottom: 1.2rem; }
        .hero-kicker { color: var(--teal); text-transform: uppercase; font-weight: 700; letter-spacing: .12em; font-size: .72rem; }
        .hero-subtitle { color: var(--muted); font-size: 1.02rem; margin: 0; }
        .section-card { background: white; border: 1px solid var(--line); border-radius: 14px; padding: 1rem 1.15rem .7rem; margin: .65rem 0; }
        .section-label { color: var(--teal-dark); font-weight: 700; font-size: .78rem; text-transform: uppercase; letter-spacing: .08em; margin-bottom: .5rem; }
        .result-card { border-radius: 14px; padding: 1.2rem 1.35rem; background: var(--mint); border-left: 6px solid var(--teal); margin-top: 1rem; }
        .result-card.alert { background: #fff0eb; border-left-color: var(--coral); }
        .result-label { color: var(--muted); font-size: .78rem; text-transform: uppercase; letter-spacing: .08em; font-weight: 700; }
        .result-value { font-family: 'Manrope', sans-serif; font-size: 1.45rem; margin: .2rem 0 .6rem; }
        .metric-card { background: white; border: 1px solid var(--line); border-radius: 12px; padding: .85rem 1rem; min-height: 104px; }
        .metric-label { color: var(--muted); font-size: .78rem; font-weight: 600; }
        .metric-value { color: var(--teal-dark); font-family: 'Manrope', sans-serif; font-size: 1.35rem; margin-top: .35rem; word-break: break-word; }
        .stButton > button { border-radius: 9px; border: 0; background: var(--teal); color: white; font-weight: 700; padding: .65rem 1.05rem; }
        .stButton > button:hover { background: var(--teal-dark); color: white; }
        [data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 10px; }
        @media (max-width: 700px) { h1 { font-size: 1.75rem; } .hero { padding: 1rem; } .metric-card { min-height: 88px; } }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_artifacts():
    required = [
        MODELS / "preprocessing_pipeline.pkl",
        MODELS / "logistic_regression.pkl",
        MODELS / "random_forest.pkl",
        MODELS / "extra_trees.pkl",
        MODELS / "hist_gradient_boosting.pkl",
    ]
    missing = [path.name for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing model artifacts: " + ", ".join(missing))
    return (
        joblib.load(required[0]),
        joblib.load(required[1]),
        joblib.load(required[2]),
        joblib.load(required[3]),
        joblib.load(required[4]),
    )


def load_report(name: str) -> pd.DataFrame:
    path = REPORTS / name
    if not path.exists():
        raise FileNotFoundError(f"Report not found: {name}")
    return pd.read_csv(path)


def candidate_form(preprocessor: RecruitmentPreprocessor) -> pd.DataFrame:
    values = {}
    st.markdown('<div class="section-card"><div class="section-label">Personal Information</div>', unsafe_allow_html=True)
    personal = st.columns(2)
    for column, feature in zip(personal, ["gender", CITY_FEATURE]):
        if feature == CITY_FEATURE:
            options = ["Unknown"] + sorted(preprocessor.city_frequency, key=preprocessor.city_frequency.get, reverse=True)[:100]
        else:
            options = list(preprocessor.encoder.categories_[NOMINAL_FEATURES.index(feature)])
        with column:
            values[feature] = st.selectbox(feature.replace("_", " ").title(), options or ["Unknown"], key=f"candidate_{feature}")
    st.markdown('</div><div class="section-card"><div class="section-label">Education & Experience</div>', unsafe_allow_html=True)
    education = st.columns(3)
    for column, feature in zip(education, ["education_level", "experience", "last_new_job"]):
        with column:
            if feature == "experience":
                values[feature] = st.selectbox("Experience", ["<1"] + [str(value) for value in range(1, 21)] + [">20"])
            else:
                values[feature] = st.selectbox(feature.replace("_", " ").title(), list(ORDINAL_MAPPINGS[feature].keys()), key=f"candidate_{feature}")
    st.markdown('</div><div class="section-card"><div class="section-label">Company Information</div>', unsafe_allow_html=True)
    company = st.columns(3)
    for column, feature in zip(company, ["relevent_experience", "enrolled_university", "major_discipline"]):
        options = list(preprocessor.encoder.categories_[NOMINAL_FEATURES.index(feature)])
        with column:
            values[feature] = st.selectbox(feature.replace("_", " ").title(), options or ["Unknown"], key=f"candidate_{feature}")
    company_more = st.columns(2)
    for column, feature in zip(company_more, ["company_size", "company_type"]):
        options = list(ORDINAL_MAPPINGS[feature].keys()) if feature in ORDINAL_FEATURES else list(preprocessor.encoder.categories_[NOMINAL_FEATURES.index(feature)])
        with column:
            values[feature] = st.selectbox(feature.replace("_", " ").title(), options or ["Unknown"], key=f"candidate_{feature}")
    st.markdown('</div><div class="section-card"><div class="section-label">Training & Development</div>', unsafe_allow_html=True)
    training = st.columns(2)
    with training[0]:
        values["city_development_index"] = st.number_input("City development index", 0.0, 1.0, 0.75, 0.01)
    with training[1]:
        values["training_hours"] = st.number_input("Training hours", 0, 500, 50, 1)
    st.markdown('</div>', unsafe_allow_html=True)
    return pd.DataFrame([values])


def main() -> None:
    st.set_page_config(page_title="Smart Recruitment Assistant", page_icon=":bar_chart:", layout="wide", initial_sidebar_state="expanded")
    inject_styles()
    st.markdown('<div class="hero"><div class="hero-kicker">AI / HR Analytics</div><h1>Smart Recruitment Assistant</h1><p class="hero-subtitle">AI-powered candidate screening and recruitment analytics.</p></div>', unsafe_allow_html=True)
    try:
        preprocessor, logistic, forest, extra_trees, gradient_boosting = load_artifacts()
    except Exception as error:
        st.error(f"The application is not ready: {error}. Run the notebook to generate artifacts.")
        st.stop()

    try:
        comparison = load_report("model_comparison.csv")
        importance = load_report("feature_importance.csv")
        coefficients = load_report("logistic_coefficients.csv")
        top_candidates = load_report("top_candidates.csv")
        insights = json.loads((REPORTS / "eda_insights.json").read_text(encoding="utf-8"))
        source_data = pd.read_csv(ROOT / "data" / "aug_train.csv")
    except Exception as error:
        st.error(f"Dashboard reports could not be loaded: {error}")
        st.stop()

    selected_name = insights.get("selected_model", "Random Forest")
    selected_model = {
        "Logistic Regression": logistic,
        "Random Forest": forest,
        "ExtraTrees": extra_trees,
        "HistGradientBoosting": gradient_boosting,
    }.get(selected_name, logistic)
    st.sidebar.markdown("<h2 style='color:#ffffff;margin-bottom:0'>Recruitment workspace</h2><p style='color:#b8d5d1'>Navigate the screening intelligence hub.</p>", unsafe_allow_html=True)
    page = st.sidebar.radio("Navigation", ["Candidate Prediction", "Recruitment Dashboard", "Top Candidates", "Model Insights", "About Project"], label_visibility="collapsed")
    st.sidebar.divider()
    st.sidebar.caption(f"Selected model\n\n**{selected_name}**")

    if page == "Candidate Prediction":
        st.header("Candidate Prediction")
        st.write("Assess one candidate using the selected model and the same preprocessing used during training.")
        candidate = candidate_form(preprocessor)
        if st.button("Predict candidate", type="primary"):
            try:
                processed = preprocessor.transform(candidate)
                if list(processed.columns) != list(preprocessor.feature_columns):
                    raise ValueError("Input features do not match the training feature schema.")
                probability = float(selected_model.predict_proba(processed)[0, 1])
                prediction = int(probability >= 0.5)
                confidence = probability if prediction else 1 - probability
                style = "" if prediction else "alert"
                st.markdown(f'<div class="result-card {style}"><div class="result-label">Screening result</div><div class="result-value">{TARGET_LABELS[prediction]}</div><div class="result-label">Confidence</div><div class="result-value">{confidence:.1%}</div></div>', unsafe_allow_html=True)
                st.info("The confidence shown is the selected model's probability for the displayed prediction. Use it as a prioritization signal, not as a standalone hiring decision.")
            except Exception as error:
                st.error(f"Prediction failed: {error}")

    elif page == "Recruitment Dashboard":
        st.header("Recruitment Dashboard")
        st.write("A report-driven view of candidate volume, model quality, and the strongest signals.")
        metrics = comparison.set_index("Model")
        total = int(insights["target_distribution"]["0.0"] + insights["target_distribution"]["1.0"])
        selected_row = comparison.loc[comparison["Model"] == selected_name].iloc[0]
        kpis = [("Dataset size", f"{total:,}"), ("Predicted job-change", f"{insights['test_predicted_job_change_count']:,}"), ("Looking for change", f"{insights['test_predicted_job_change_percentage']:.1f}%"), ("Selected model", selected_name), ("Accuracy", f"{selected_row['Accuracy']:.1%}"), ("F1 score", f"{selected_row['F1']:.1%}"), ("ROC-AUC", f"{selected_row['ROC-AUC']:.1%}")]
        for start in range(0, len(kpis), 4):
            row = st.columns(min(4, len(kpis) - start))
            for column, (label, value) in zip(row, kpis[start:start + 4]):
                column.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>', unsafe_allow_html=True)
        left, right = st.columns(2)
        with left:
            st.subheader("Model comparison")
            st.bar_chart(metrics[["Accuracy", "Precision", "Recall", "F1", "Balanced Accuracy", "ROC-AUC"]])
        with right:
            st.subheader("Feature importance")
            st.bar_chart(importance.head(12).set_index("Feature")["Importance"])
        st.subheader("Candidate profile")
        profile = st.columns(3)
        profile[0].bar_chart(source_data["education_level"].fillna("Unknown").value_counts())
        profile[1].bar_chart(source_data["experience"].fillna("Unknown").value_counts())
        profile[2].line_chart(source_data["training_hours"].value_counts().sort_index())

    elif page == "Top Candidates":
        st.header("Top Candidates")
        st.write("Highest-confidence test-set records from the selected model.")
        display_columns = [column for column in [ID_COLUMN, "Prediction", "Confidence", "relevent_experience", "education_level", "experience", "company_size", "last_new_job", "training_hours"] if column in top_candidates]
        shown = top_candidates[display_columns].copy()
        shown.insert(0, "Rank", range(1, len(shown) + 1))
        shown["Prediction"] = shown["Prediction"].map(TARGET_LABELS)
        shown["Confidence"] = shown["Confidence"].map(lambda value: f"{value:.1%}")
        st.dataframe(shown, use_container_width=True, hide_index=True)
        st.subheader("Confidence by rank")
        confidence_chart = top_candidates[[ID_COLUMN, "Confidence"]].copy()
        confidence_chart.index = range(1, len(confidence_chart) + 1)
        st.bar_chart(confidence_chart["Confidence"])

    elif page == "Model Insights":
        st.header("Model Insights")
        st.write("Metrics are calculated on the held-out test set. Higher recall helps reduce missed interested candidates; F1 balances recall and precision.")
        st.dataframe(comparison[["Model", "Accuracy", "Precision", "Recall", "F1", "Balanced Accuracy", "ROC-AUC"]].style.format({column: "{:.1%}" for column in comparison.columns if column != "Model"}), use_container_width=True, hide_index=True)
        st.success(f"Selected model: {selected_name}")
        st.subheader("Logistic Regression: positive signals")
        st.dataframe(coefficients.sort_values("Coefficient", ascending=False).head(10), hide_index=True, use_container_width=True)
        st.subheader("Logistic Regression: negative signals")
        st.dataframe(coefficients.sort_values("Coefficient").head(10), hide_index=True, use_container_width=True)
        st.subheader("Selected-model feature importance")
        st.dataframe(importance.head(15), hide_index=True, use_container_width=True)
        st.write("Positive logistic coefficients increase estimated job-change likelihood; negative coefficients decrease it. Selected-model importance indicates predictive contribution, not causation.")

    else:
        st.header("About Project")
        about = {
            "Problem": "Help recruitment teams prioritize candidates who may be looking for a job change.",
            "Dataset": "HR Analytics Job Change data with candidate education, experience, company, city, and training fields.",
            "Preprocessing": "Training-only imputation, ordinal encoding, city frequency encoding, one-hot encoding, and scaling.",
            "Models": "Logistic Regression, Random Forest, ExtraTrees, and HistGradientBoosting.",
            "Evaluation": "Accuracy, Precision, Recall, F1, Balanced Accuracy, and ROC-AUC on a fixed stratified holdout.",
            "Dashboard": "A report-driven Streamlit workspace for prediction, ranking, model review, and recruitment analytics.",
            "Project goal": "Support consistent screening decisions while keeping model limitations and human judgment visible.",
        }
        for label, text in about.items():
            st.markdown(f'<div class="section-card"><div class="section-label">{label}</div><div>{text}</div></div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
