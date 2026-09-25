"""Shared helpers reused across every notebook in this project.

Keeping these in one place means every notebook plots and scores things the
same way, and a fix here (e.g. a metric definition) doesn't need to be
copy-pasted into five different files.
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report,
)


def project_root():
    for p in (Path.cwd(), Path.cwd().parent):
        if (p / "data" / "raw").exists():
            return p
    raise FileNotFoundError("Could not find the project root (looked for a data/raw folder).")


ROOT = project_root()
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
FIG_DIR = ROOT / "outputs" / "figures"
RESULTS_DIR = ROOT / "outputs" / "results"
for _d in (PROC_DIR, FIG_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

CLASS_ORDER = ["Low", "Medium", "High"]


# --------------------------------------------------------------------------
# General-purpose plots (used from preprocessing through to evaluation)
# --------------------------------------------------------------------------

def quick_summary(df, name):
    """Shape, preview and missingness -- our standard first look at any dataframe."""
    print(f"--- {name} ---")
    print("shape:", df.shape)
    display(df.head(3))
    missing = df.isna().mean().sort_values(ascending=False)
    missing = missing[missing > 0]
    if len(missing):
        print("columns with missing values (%):")
        display((missing * 100).round(2))


def plot_missingness(df, title, ax=None):
    pct = df.isna().mean().sort_values(ascending=False) * 100
    pct = pct[pct > 0]
    ax = ax or plt.gca()
    if pct.empty:
        ax.text(0.5, 0.5, "No missing values", ha="center")
    else:
        pct.plot(kind="barh", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("% missing")
    return ax


def plot_hist(series, title, xlabel, bins=30, ax=None):
    ax = ax or plt.gca()
    series.dropna().plot(kind="hist", bins=bins, ax=ax, edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    return ax


def plot_boxplot_by_class(df, feature, class_col="target_exposure_class", class_order=CLASS_ORDER,
                           title=None, ax=None):
    """One feature's spread across Low/Medium/High -- the standard check for
    whether a feature actually separates the classes at all."""
    ax = ax or plt.gca()
    data = [df.loc[df[class_col] == c, feature].dropna() for c in class_order]
    ax.boxplot(data, tick_labels=class_order)
    ax.set_title(title or feature)
    ax.grid(axis="y", alpha=0.3)
    return ax


def plot_class_counts(train_series, test_series, class_order=CLASS_ORDER, title="Class distribution", ax=None):
    counts = pd.DataFrame({
        "Training": train_series.value_counts().reindex(class_order),
        "Testing": test_series.value_counts().reindex(class_order),
    })
    ax = counts.plot(kind="bar", rot=0, ax=ax)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    return ax


# --------------------------------------------------------------------------
# Model evaluation (used identically for every one of the four models)
# --------------------------------------------------------------------------

def evaluate_classifier(model_name, y_true, y_pred, class_order=CLASS_ORDER, verbose=True):
    """The exact metric set the assessment asks for: Accuracy, Precision, Recall, F1.

    Both macro (every class weighted equally) and weighted (by class size) averages
    are reported, because the test set is not perfectly class-balanced -- macro
    tells you how well the model handles the smallest class, weighted tells you
    overall performance as actually distributed in the test set.
    """
    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    if verbose:
        print(f"=== {model_name} ===")
        print(classification_report(y_true, y_pred, labels=class_order, zero_division=0))
    return metrics


def plot_confusion(y_true, y_pred, class_order=CLASS_ORDER, title="Confusion matrix", ax=None):
    cm = confusion_matrix(y_true, y_pred, labels=class_order)
    ax = ax or plt.gca()
    ConfusionMatrixDisplay(cm, display_labels=class_order).plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(title)
    return ax


def save_metrics(metrics: dict, filename="model_metrics.json"):
    """Each model's metrics are merged into one shared JSON file, keyed by model
    name, so the comparison notebook can load every result without re-running
    any model."""
    path = RESULTS_DIR / filename
    existing = json.loads(path.read_text()) if path.exists() else {}
    existing[metrics["model"]] = metrics
    path.write_text(json.dumps(existing, indent=2))
    print(f"Saved metrics for '{metrics['model']}' to {path}")


def load_all_metrics(filename="model_metrics.json"):
    path = RESULTS_DIR / filename
    if not path.exists():
        return {}
    return json.loads(path.read_text())
