import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

from src.data_preprocessing import prepare_data
from src.evaluate import (
    evaluate_model,
    print_metrics,
    plot_confusion_matrix,
    plot_roc_curve
)


def main():

    # -------------------------
    # 1. Load and clean data
    # -------------------------

    df = prepare_data()

    print(
        "Dataset shape:",
        df.shape
    )

    # -------------------------
    # 2. Prepare features
    # -------------------------

    X = df.drop(
        columns=["TYPE"]
    )

    y = df["TYPE"]

    # -------------------------
    # 3. Train-test split
    # -------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # -------------------------
    # 4. Base model
    # -------------------------

    base_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42
    )

    # -------------------------
    # 5. Calibration
    # -------------------------

    model = CalibratedClassifierCV(
        base_model,
        method="sigmoid",
        cv=5
    )

    model.fit(
        X_train,
        y_train
    )

    # -------------------------
    # 6. Evaluation
    # -------------------------

    metrics, y_pred, y_prob = evaluate_model(
        model,
        X_test,
        y_test
    )

    print_metrics(metrics)

    # -------------------------
    # 7. Visualizations
    # -------------------------

    plot_confusion_matrix(
        y_test,
        y_pred
    )

    plot_roc_curve(
        y_test,
        y_prob
    )


if __name__ == "__main__":
    main()
