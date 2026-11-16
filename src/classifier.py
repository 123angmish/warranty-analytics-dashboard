"""
src/classifier.py
-----------------
NLP Failure Mode Classifier for Warranty Free-Text Claim Descriptions.
Pipelines:
  - TF-IDF Vectorizer (sublinear TF, n-grams (1, 2), min_df=2)
  - Classifier comparison: LogisticRegression vs LinearSVC vs MultinomialNB
  - Honest evaluation: Stratified Train / Validation / Test split, 5-Fold Cross-Validation,
    Macro-F1, Weighted-F1, Accuracy, Confusion Matrix, Classification Report.
  - Serialization with joblib -> models/failure_mode_pipeline.joblib
  - Metrics export -> reports/metrics.json
"""

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports")
MODEL_PATH = os.path.join(MODELS_DIR, "failure_mode_pipeline.joblib")
METRICS_JSON_PATH = os.path.join(REPORTS_DIR, "metrics.json")


def preprocess_text(text: str) -> str:
    """Cleans raw claim description text while retaining electrical acronyms and terms."""
    if not isinstance(text, str):
        return ""
    # Lowercase
    t = text.lower()
    # Normalize whitespace and symbols
    t = re.sub(r"[^\w\s\-\/\.]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def train_and_evaluate_models(
    claims_df: Optional[pd.DataFrame] = None,
    save_artifacts: bool = True,
) -> Dict[str, Any]:
    """
    Trains and compares multiple NLP classifiers on cleaned claim descriptions.
    Generates realistic evaluation metrics and serializes the champion model.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    if claims_df is None:
        clean_csv = os.path.join(DATA_DIR, "claims_clean.csv")
        if not os.path.exists(clean_csv):
            raise FileNotFoundError(f"Missing {clean_csv}. Please run src.db first.")
        claims_df = pd.read_csv(clean_csv)

    print(f"Loaded {len(claims_df):,} claims for NLP model training & evaluation...")

    # Ensure required columns exist
    df = claims_df[["claim_description", "failure_mode", "product_family"]].dropna().copy()
    df["clean_text"] = df["claim_description"].apply(preprocess_text)

    X = df["clean_text"].values
    y = df["failure_mode"].values
    classes = sorted(list(np.unique(y)))

    # Stratified Split: 70% Train, 15% Validation, 15% Test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.1765, random_state=42, stratify=y_train_val  # 0.1765 of 85% ~ 15% of total
    )

    print(f"Dataset Split Sizes: Train={len(X_train):,}, Validation={len(X_val):,}, Test={len(X_test):,}")

    # Models to benchmark
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=4000,
        sublinear_tf=True,
        min_df=2,
    )

    candidate_models = {
        "LogisticRegression": LogisticRegression(
            C=2.5,
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
            solver="lbfgs",
        ),
        "LinearSVC": LinearSVC(
            C=1.0,
            class_weight="balanced",
            random_state=42,
            dual="auto",
        ),
        "MultinomialNB": MultinomialNB(
            alpha=0.5,
        ),
    }

    comparison_results = []
    trained_pipelines = {}

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, clf in candidate_models.items():
        pipe = Pipeline([
            ("tfidf", vectorizer),
            ("clf", clf),
        ])

        print(f"Cross-validating and fitting {name}...")
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1)
        mean_cv_macro_f1 = float(np.mean(cv_scores))

        # Fit on Train
        pipe.fit(X_train, y_train)

        # Evaluate on Validation
        val_preds = pipe.predict(X_val)
        val_acc = float(accuracy_score(y_val, val_preds))
        val_macro_f1 = float(f1_score(y_val, val_preds, average="macro"))
        val_weighted_f1 = float(f1_score(y_val, val_preds, average="weighted"))

        comparison_results.append({
            "model_name": name,
            "cv_macro_f1": round(mean_cv_macro_f1, 4),
            "val_accuracy": round(val_acc, 4),
            "val_macro_f1": round(val_macro_f1, 4),
            "val_weighted_f1": round(val_weighted_f1, 4),
        })
        trained_pipelines[name] = pipe

    comp_df = pd.DataFrame(comparison_results)
    print("\nModel Comparison Table on Validation Set:")
    print(comp_df.to_string(index=False))

    # Champion model selection (LogisticRegression chosen as champion for probabilistic inference)
    champion_name = "LogisticRegression"
    champion_pipe = trained_pipelines[champion_name]

    # Refit champion on Train + Val for final Test set evaluation
    X_train_full = np.concatenate([X_train, X_val])
    y_train_full = np.concatenate([y_train, y_val])
    champion_pipe.fit(X_train_full, y_train_full)

    # Final hold-out Test Set Evaluation
    test_preds = champion_pipe.predict(X_test)
    test_acc = float(accuracy_score(y_test, test_preds))
    test_macro_f1 = float(f1_score(y_test, test_preds, average="macro"))
    test_weighted_f1 = float(f1_score(y_test, test_preds, average="weighted"))
    test_precision = float(precision_score(y_test, test_preds, average="macro"))
    test_recall = float(recall_score(y_test, test_preds, average="macro"))

    report_dict = classification_report(y_test, test_preds, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, test_preds, labels=classes)

    print(f"\nChampion Model [{champion_name}] Final Test Metrics:")
    print(f"  Accuracy:    {test_acc * 100:.2f}%")
    print(f"  Macro-F1:    {test_macro_f1 * 100:.2f}%")
    print(f"  Weighted-F1: {test_weighted_f1 * 100:.2f}%")
    print(f"  Precision:   {test_precision * 100:.2f}%")
    print(f"  Recall:      {test_recall * 100:.2f}%")

    # Incorporate Weibull reliability results if available
    weibull_summary_dict = []
    try:
        from src.reliability import fit_all_products_weibull
        sales_csv = os.path.join(DATA_DIR, "sales.csv")
        if os.path.exists(sales_csv):
            sales_df = pd.read_csv(sales_csv)
            w_df = fit_all_products_weibull(claims_df, sales_df)
            weibull_summary_dict = w_df.to_dict(orient="records")
    except Exception as e:
        print(f"Warning: could not compute Weibull metrics for json: {e}")

    # Metadata metrics dictionary
    final_metrics = {
        "model_champion": champion_name,
        "test_accuracy": round(test_acc, 4),
        "test_macro_f1": round(test_macro_f1, 4),
        "test_weighted_f1": round(test_weighted_f1, 4),
        "test_precision": round(test_precision, 4),
        "test_recall": round(test_recall, 4),
        "num_records_total": len(claims_df),
        "num_records_train": len(X_train),
        "num_records_val": len(X_val),
        "num_records_test": len(X_test),
        "num_product_families": int(claims_df["product_family"].nunique()),
        "num_failure_modes": len(classes),
        "classes": classes,
        "model_comparison": comparison_results,
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist(),
        "weibull_reliability": weibull_summary_dict,
    }

    if save_artifacts:
        joblib.dump(champion_pipe, MODEL_PATH)
        print(f"Saved trained champion pipeline -> {MODEL_PATH}")

        with open(METRICS_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(final_metrics, f, indent=2)
        print(f"Saved evaluation metrics -> {METRICS_JSON_PATH}")

    return {
        "pipeline": champion_pipe,
        "metrics": final_metrics,
        "comparison_df": comp_df,
    }


def load_classifier_pipeline(model_path: str = MODEL_PATH) -> Pipeline:
    """Loads the serialized TF-IDF + Classifier pipeline."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Train model first.")
    return joblib.load(model_path)


def predict_failure_mode(
    text: str,
    pipeline: Optional[Pipeline] = None,
    top_k: int = 3,
) -> Dict[str, Any]:
    """
    Predicts the failure mode for a given free-text warranty claim description.

    Returns:
        Dictionary with:
          - 'predicted_failure_mode': Most likely class
          - 'confidence': Probability score of top class (0.0 to 1.0)
          - 'top_k_predictions': List of dicts with failure mode and probability
          - 'cleaned_input': Preprocessed text
    """
    if pipeline is None:
        pipeline = load_classifier_pipeline()

    clean_input = preprocess_text(text)
    if not clean_input:
        return {
            "predicted_failure_mode": "Unknown",
            "confidence": 0.0,
            "top_k_predictions": [],
            "cleaned_input": clean_input,
        }

    # If classifier supports predict_proba
    clf = pipeline.named_steps.get("clf")
    classes = pipeline.classes_

    if hasattr(clf, "predict_proba"):
        probs = pipeline.predict_proba([clean_input])[0]
        top_indices = np.argsort(probs)[::-1][:top_k]
        top_k_list = [
            {"failure_mode": str(classes[idx]), "probability": round(float(probs[idx]), 4)}
            for idx in top_indices
        ]
        champion_label = str(classes[top_indices[0]])
        champion_conf = round(float(probs[top_indices[0]]), 4)
    else:
        # Decision function fallback
        dec = pipeline.decision_function([clean_input])[0]
        pred_label = pipeline.predict([clean_input])[0]
        champion_label = str(pred_label)
        champion_conf = 1.0
        top_k_list = [{"failure_mode": champion_label, "probability": 1.0}]

    return {
        "predicted_failure_mode": champion_label,
        "confidence": champion_conf,
        "top_k_predictions": top_k_list,
        "cleaned_input": clean_input,
    }


def main():
    print("=" * 60)
    print("TRAINING & EVALUATING NLP WARRANTY FAILURE MODE CLASSIFIER")
    print("=" * 60)
    res = train_and_evaluate_models()
    metrics = res["metrics"]
    print("\nSummary of Trained Model Artifacts:")
    print(f"  Model Saved:       {MODEL_PATH}")
    print(f"  Metrics JSON:      {METRICS_JSON_PATH}")
    print(f"  Overall Accuracy:  {metrics['test_accuracy'] * 100:.2f}%")
    print(f"  Macro-F1 Score:    {metrics['test_macro_f1'] * 100:.2f}%")
    print(f"  Weighted-F1 Score: {metrics['test_weighted_f1'] * 100:.2f}%")

    # Sample interactive inference test
    sample_texts = [
        "Capacitor bank bulged with heavy electrolyte leakage and charred vent seal on DC bus.",
        "ACB failed to open during annual secondary trip test; shunt trip coil showed open circuit.",
        "High humidity near coast caused severe galvanic corrosion and green oxidation on copper busbars.",
        "Cooling fan seized up and logged 92C thermal fault code E-04 on inverter bridge.",
    ]
    print("\nTesting Model Predictions on Sample Claim Notes:")
    for txt in sample_texts:
        pred = predict_failure_mode(txt, pipeline=res["pipeline"])
        print(f"  Input: \"{txt[:60]}...\"")
        print(f"  --> Predicted: {pred['predicted_failure_mode']} ({pred['confidence']*100:.1f}% confidence)")


if __name__ == "__main__":
    main()
