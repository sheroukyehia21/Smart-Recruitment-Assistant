from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight


TARGET_COLUMN = "target"
ID_COLUMN = "enrollee_id"
NUMERICAL_FEATURES = ["city_development_index", "training_hours", "experience"]
ORDINAL_FEATURES = ["education_level", "company_size", "last_new_job"]
NOMINAL_FEATURES = [
    "gender",
    "relevent_experience",
    "enrolled_university",
    "major_discipline",
    "company_type",
]
CITY_FEATURE = "city"
MODEL_FEATURE_COLUMNS = [
    "city_development_index",
    "training_hours",
    "experience",
    "education_level",
    "company_size",
    "last_new_job",
    "city_freq",
]

ORDINAL_MAPPINGS = {
    "education_level": {
        "Unknown": -1, "Primary School": 0, "High School": 1,
        "Graduate": 2, "Masters": 3, "Phd": 4,
    },
    "company_size": {
        "Unknown": -1, "<10": 0, "10/49": 1, "50-99": 2,
        "100-500": 3, "500-999": 4, "1000-4999": 5,
        "5000-9999": 6, "10000+": 7,
    },
    "last_new_job": {
        "Unknown": -1, "never": 0, "1": 1, "2": 2,
        "3": 3, "4": 4, ">4": 5,
    },
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_experience(value: object) -> float:
    if pd.isna(value):
        return float("nan")
    value = str(value).strip()
    if value == ">20":
        return 21.0
    if value == "<1":
        return 0.0
    try:
        return float(value)
    except ValueError:
        return float("nan")


class RecruitmentPreprocessor:
    """Fit all transformations on training data and reuse them at inference."""

    def __init__(self) -> None:
        self.numeric_imputer = SimpleImputer(strategy="median")
        self.categorical_imputer = SimpleImputer(strategy="constant", fill_value="Unknown")
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.scaler = StandardScaler()
        self.city_frequency: dict[str, float] = {}
        self.feature_columns: list[str] = []

    def _prepare(self, frame: pd.DataFrame) -> pd.DataFrame:
        prepared = frame.copy()
        prepared["experience"] = prepared["experience"].map(parse_experience)
        prepared[NUMERICAL_FEATURES] = self.numeric_imputer.transform(prepared[NUMERICAL_FEATURES])
        categorical = ORDINAL_FEATURES + NOMINAL_FEATURES + [CITY_FEATURE]
        prepared[categorical] = self.categorical_imputer.transform(prepared[categorical])
        for column, mapping in ORDINAL_MAPPINGS.items():
            prepared[column] = prepared[column].map(mapping).fillna(-1)
        prepared["city_freq"] = prepared[CITY_FEATURE].map(self.city_frequency).fillna(0.0)
        return prepared

    def fit(self, frame: pd.DataFrame) -> "RecruitmentPreprocessor":
        prepared = frame.copy()
        prepared["experience"] = prepared["experience"].map(parse_experience)
        self.numeric_imputer.fit(prepared[NUMERICAL_FEATURES])
        categorical = ORDINAL_FEATURES + NOMINAL_FEATURES + [CITY_FEATURE]
        self.categorical_imputer.fit(prepared[categorical])
        filled = prepared.copy()
        filled[NUMERICAL_FEATURES] = self.numeric_imputer.transform(filled[NUMERICAL_FEATURES])
        filled[categorical] = self.categorical_imputer.transform(filled[categorical])
        self.city_frequency = filled[CITY_FEATURE].value_counts(normalize=True).to_dict()
        self.encoder.fit(filled[NOMINAL_FEATURES])
        base = filled[MODEL_FEATURE_COLUMNS[:-1]].copy()
        for column, mapping in ORDINAL_MAPPINGS.items():
            base[column] = base[column].map(mapping).fillna(-1)
        base["city_freq"] = filled[CITY_FEATURE].map(self.city_frequency).fillna(0.0)
        self.scaler.fit(base[NUMERICAL_FEATURES + ["city_freq"]])
        self.feature_columns = MODEL_FEATURE_COLUMNS + list(self.encoder.get_feature_names_out(NOMINAL_FEATURES))
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        prepared = self._prepare(frame)
        base = prepared[MODEL_FEATURE_COLUMNS].copy()
        base[NUMERICAL_FEATURES + ["city_freq"]] = self.scaler.transform(
            base[NUMERICAL_FEATURES + ["city_freq"]]
        )
        encoded = pd.DataFrame(
            self.encoder.transform(prepared[NOMINAL_FEATURES]),
            columns=self.encoder.get_feature_names_out(NOMINAL_FEATURES),
            index=prepared.index,
        )
        result = pd.concat([base, encoded], axis=1)
        return result.reindex(columns=self.feature_columns, fill_value=0.0)


def calculate_metrics(model, features: pd.DataFrame, target: pd.Series) -> tuple[dict, pd.DataFrame]:
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1]
    metrics = {
        "Accuracy": accuracy_score(target, predictions),
        "Precision": precision_score(target, predictions, zero_division=0),
        "Recall": recall_score(target, predictions, zero_division=0),
        "F1": f1_score(target, predictions, zero_division=0),
        "Balanced Accuracy": balanced_accuracy_score(target, predictions),
        "ROC-AUC": roc_auc_score(target, probabilities),
    }
    report = pd.DataFrame(classification_report(target, predictions, output_dict=True, zero_division=0)).T
    return metrics, report


def train_and_export(data_path: Path | None = None, output_root: Path | None = None) -> dict:
    root = output_root or project_root()
    data_path = data_path or root / "data" / "aug_train.csv"
    models_dir, reports_dir = root / "models", root / "reports"
    models_dir.mkdir(exist_ok=True)
    reports_dir.mkdir(exist_ok=True)

    data = pd.read_csv(data_path)
    X_raw = data.drop(columns=[TARGET_COLUMN])
    y = data[TARGET_COLUMN].astype(int)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=42, stratify=y
    )
    preprocessor = RecruitmentPreprocessor().fit(X_train_raw)
    X_train = preprocessor.transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)
    if list(X_train.columns) != list(X_test.columns) or X_train.isna().any().any() or X_test.isna().any().any():
        raise ValueError("Preprocessing produced mismatched columns or unexpected NaN values.")

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=42, class_weight="balanced", n_jobs=-1
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.05, max_leaf_nodes=31, random_state=42
        ),
    }
    results = []
    reports = {}
    for name, model in models.items():
        if name == "HistGradientBoosting":
            model.fit(X_train, y_train, sample_weight=compute_sample_weight("balanced", y_train))
        else:
            model.fit(X_train, y_train)
        metrics, report = calculate_metrics(model, X_test, y_test)
        results.append({"Model": name, **metrics})
        reports[name] = report

    comparison = pd.DataFrame(results)
    selection_metrics = ["Recall", "F1", "Balanced Accuracy", "ROC-AUC"]
    comparison["Selection Score"] = comparison[selection_metrics].mean(axis=1)
    selected_name = comparison.sort_values("Selection Score", ascending=False).iloc[0]["Model"]
    selected_model = models[selected_name]
    joblib.dump(preprocessor, models_dir / "preprocessing_pipeline.pkl")
    joblib.dump(models["Logistic Regression"], models_dir / "logistic_regression.pkl")
    joblib.dump(models["Random Forest"], models_dir / "random_forest.pkl")
    joblib.dump(models["ExtraTrees"], models_dir / "extra_trees.pkl")
    joblib.dump(models["HistGradientBoosting"], models_dir / "hist_gradient_boosting.pkl")
    comparison.to_csv(reports_dir / "model_comparison.csv", index=False)

    coefficients = pd.DataFrame({"Feature": X_train.columns, "Coefficient": models["Logistic Regression"].coef_[0]})
    coefficients["Absolute_Coefficient"] = coefficients["Coefficient"].abs()
    coefficients.sort_values("Absolute_Coefficient", ascending=False).to_csv(
        reports_dir / "logistic_coefficients.csv", index=False
    )
    if hasattr(selected_model, "feature_importances_"):
        selected_importance = selected_model.feature_importances_
        importance_method = "native feature importance"
    else:
        permutation = permutation_importance(
            selected_model,
            X_test,
            y_test,
            n_repeats=5,
            random_state=42,
            scoring="roc_auc",
            n_jobs=-1,
        )
        selected_importance = permutation.importances_mean
        importance_method = "permutation importance (ROC-AUC)"
    importance = pd.DataFrame({"Feature": X_train.columns, "Importance": selected_importance})
    importance["Method"] = importance_method
    importance.sort_values("Importance", ascending=False).to_csv(reports_dir / "feature_importance.csv", index=False)

    probabilities = selected_model.predict_proba(X_test)[:, 1]
    top = X_test_raw.copy().reset_index(drop=True)
    top["Prediction"] = (probabilities >= 0.5).astype(int)
    top["Confidence"] = probabilities
    top = top.assign(_rank_score=probabilities).sort_values("_rank_score", ascending=False).head(10).drop(columns="_rank_score")
    candidate_columns = [
        column for column in [
            ID_COLUMN, "Prediction", "Confidence", "relevent_experience",
            "education_level", "experience", "company_size", "last_new_job",
            "training_hours",
        ] if column in top.columns
    ]
    top = top[candidate_columns]
    top.to_csv(reports_dir / "top_candidates.csv", index=False)
    test_predictions = selected_model.predict(X_test)

    eda_insights = {
        "target_distribution": data[TARGET_COLUMN].value_counts().sort_index().astype(int).to_dict(),
        "target_percentages": (data[TARGET_COLUMN].value_counts(normalize=True).sort_index() * 100).round(2).to_dict(),
        "selected_model": selected_name,
        "selection_rule": "Highest mean of Recall, F1, Balanced Accuracy, and ROC-AUC.",
        "test_record_count": int(len(X_test)),
        "test_predicted_job_change_count": int(test_predictions.sum()),
        "test_predicted_job_change_percentage": round(float(test_predictions.mean() * 100), 2),
        "education_distribution": data["education_level"].fillna("Unknown").value_counts().to_dict(),
        "experience_distribution": data["experience"].fillna("Unknown").value_counts().to_dict(),
        "training_hours_summary": data["training_hours"].describe().round(2).to_dict(),
    }
    (reports_dir / "eda_insights.json").write_text(json.dumps(eda_insights, indent=2), encoding="utf-8")
    return {"comparison": comparison, "selected_model": selected_name, "reports": reports, "test_raw": X_test_raw}
