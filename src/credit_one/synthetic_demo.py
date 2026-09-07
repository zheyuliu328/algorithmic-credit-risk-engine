"""Fresh offline classification with no business-data calibration.

Independent normal features generate logistic latent event probabilities and
Bernoulli labels. No legacy scoring code, client identifiers or overrides are used.
"""

import platform

import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


COEFFICIENTS = (0.8, -0.6, 0.4, 0.0)
INTERCEPT = -1.0


def generate_dataset(seed=42, n_samples=1000):
    """Return independent synthetic features and stochastic binary event labels."""
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer between 0 and 2**32 - 1")
    if not isinstance(n_samples, (int, np.integer)) or isinstance(n_samples, bool) or n_samples < 100:
        raise ValueError("n_samples must be an integer of at least 100")
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(n_samples, len(COEFFICIENTS)))
    logits = INTERCEPT + features @ np.asarray(COEFFICIENTS)
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    labels = rng.binomial(1, probabilities)
    if np.unique(labels).size != 2 or np.bincount(labels).min() < 2:
        raise ValueError("Both classes need at least two samples; no automatic seed retry is used")
    return features, labels


def classification_metrics(labels, probabilities):
    """Compute binary AUC, two-sided KS separation and binary Brier loss."""
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities, dtype=float)
    if labels.ndim != 1 or probabilities.ndim != 1 or labels.shape != probabilities.shape:
        raise ValueError("labels and probabilities must be equally sized one-dimensional arrays")
    if not np.array_equal(np.unique(labels), np.array([0, 1])):
        raise ValueError("Both binary label classes 0 and 1 are required")
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Probabilities must be finite and within [0, 1]")
    fpr, tpr, _ = roc_curve(labels, probabilities, drop_intermediate=False)
    return {
        "auc": float(roc_auc_score(labels, probabilities)),
        "ks": float(np.max(np.abs(tpr - fpr))),
        "brier": float(brier_score_loss(labels, probabilities)),
    }


def run_experiment(seed=42, n_samples=1000):
    """Fit once; evaluate the untouched stratified holdout without tuning."""
    features, labels = generate_dataset(seed, n_samples)
    train_indices, test_indices = train_test_split(
        np.arange(n_samples), test_size=0.25, random_state=seed, stratify=labels
    )
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=seed)),
    ])
    model.fit(features[train_indices], labels[train_indices])
    probabilities = model.predict_proba(features[test_indices])[:, 1]
    test_labels = labels[test_indices]
    training_prevalence = float(labels[train_indices].mean())
    scaler = model.named_steps["scaler"]
    classifier = model.named_steps["classifier"]
    return {
        "schema_version": 1,
        "mode": "SYNTHETIC_CLASSIFICATION_DEMO",
        "model": "StandardScaler + LogisticRegression",
        "seed": int(seed),
        "n_samples": int(n_samples),
        "data_generation": {
            "source": "Independent standard-normal features; no external data",
            "n_features": len(COEFFICIENTS),
            "latent_logistic_intercept": INTERCEPT,
            "latent_logistic_coefficients": list(COEFFICIENTS),
            "labels": "Bernoulli draws from the latent logistic probabilities",
        },
        "split": {
            "method": "Stratified random 75/25 split for IID rows; not out-of-time validation",
            "train_samples": len(train_indices),
            "test_samples": len(test_indices),
            "train_indices": train_indices.tolist(),
            "test_indices": test_indices.tolist(),
        },
        "fit": {
            "C": 1.0,
            "solver": "lbfgs",
            "max_iter": 1000,
            "scaler_fit_samples": int(scaler.n_samples_seen_),
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "coefficients_on_scaled_features": classifier.coef_[0].tolist(),
            "intercept_on_scaled_features": float(classifier.intercept_[0]),
        },
        "metrics": classification_metrics(test_labels, probabilities),
        "reference": {
            "model": "Constant training-event-rate predictor",
            "probability": training_prevalence,
            "metrics": classification_metrics(
                test_labels, np.full(len(test_labels), training_prevalence)
            ),
        },
        "held_out": [
            {"sample_index": int(index), "label": int(label), "probability": float(probability)}
            for index, label, probability in zip(test_indices, test_labels, probabilities)
        ],
        "metric_definitions": {
            "auc": "Held-out ROC AUC; higher is better",
            "ks": "Maximum absolute (TPR - FPR) across ROC thresholds; separation, not calibration",
            "brier": "Mean squared probability error on binary labels; lower is better",
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "limitations": [
            "Arbitrary simulation coefficients are not calibrated to credit or borrower data.",
            "One IID holdout does not establish temporal stability, fairness or regulatory compliance.",
            "No seed, feature or hyperparameter search is performed; scores need not beat the reference.",
            "Reproduction assumes compatible numerical libraries; versions and holdout predictions are saved.",
        ],
    }
