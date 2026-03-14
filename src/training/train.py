import os
import sys
import joblib
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
load_dotenv()

from src.data.loader import load_raw_data, basic_clean
from src.data.validator import validate
from src.features.engineer import create_domain_features
from src.models.pipeline import build_pipeline
from src.models.evaluator import (
    find_optimal_threshold,
    evaluate_model,
    compare_thresholds,
)
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__, log_file="logs/training.log")


def run_cross_validation(pipeline, X_train, y_train) -> dict:
    logger.info("Running 5-fold stratified cross validation...")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scoring = {
        "roc_auc": "roc_auc",
        "f1": "f1",
        "precision": "precision",
        "recall": "recall",
        "avg_precision": "average_precision",
    }

    results = cross_validate(
        pipeline, X_train, y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
    )

    # Summarize
    summary = {}
    for key, values in results.items():
        if key.startswith("test_"):
            metric = key.replace("test_", "")
            summary[metric] = {
                "mean": round(values.mean(), 4),
                "std": round(values.std(), 4),
            }
            logger.info(f"  CV {metric}: {values.mean():.4f} ± {values.std():.4f}")

    return summary


def main():
    config = load_config()

    # ── 1. Data ──────────────────────────────────────────
    df = load_raw_data()
    df = basic_clean(df)
    validate(df)
    df = create_domain_features(df)

    X = df.drop(columns=["Churn", "customerID"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["data"]["test_size"],
        stratify=y,
        random_state=config["data"]["random_state"],
    )
    logger.info(f"Train: {X_train.shape} | Test: {X_test.shape}")
    logger.info(f"Train churn rate: {y_train.mean():.2%}")
    logger.info(f"Test churn rate:  {y_test.mean():.2%}")

    # ── 2. Cross Validation ──────────────────────────────
    pipeline = build_pipeline(config)
    cv_results = run_cross_validation(pipeline, X_train, y_train)

    # ── 3. Final Training ────────────────────────────────
    logger.info("Fitting final pipeline on full training set...")
    pipeline.fit(X_train, y_train)
    logger.info("Training complete")

    # ── 4. Threshold Tuning ──────────────────────────────
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    # Show all thresholds so you understand the tradeoff
    compare_thresholds(y_test.values, y_proba)

    # Find optimal threshold
    optimal_threshold, best_f1 = find_optimal_threshold(
        y_test.values, y_proba,
        metric="f1",
        search_range=(0.2, 0.6),
        steps=41,
    )

    # ── 5. Final Evaluation ──────────────────────────────
    metrics = evaluate_model(y_test.values, y_proba, optimal_threshold)

    # ── 6. Save Artifacts ────────────────────────────────
    artifacts_path = Path(os.getenv("ARTIFACTS_PATH", "artifacts/"))
    artifacts_path.mkdir(exist_ok=True)

    model_path = artifacts_path / "churn_pipeline.joblib"
    threshold_path = artifacts_path / "threshold.txt"

    joblib.dump(pipeline, model_path)
    threshold_path.write_text(str(optimal_threshold))

    logger.info(f"Model saved:     {model_path}")
    logger.info(f"Threshold saved: {threshold_path}")
    logger.info(f"Final ROC-AUC:   {metrics['roc_auc']}")
    logger.info(f"Final F1:        {metrics['f1']}")
    logger.info(f"Churners caught: {metrics['churn_caught_rate']:.2%}")

    return pipeline, metrics, cv_results, optimal_threshold


if __name__ == "__main__":
    main()