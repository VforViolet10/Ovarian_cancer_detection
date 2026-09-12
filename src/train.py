import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


def load_dataset():

    df = pd.read_csv(
        "data/processed/data.csv"
    )

    X = df.drop(columns=["TYPE"])
    y = df["TYPE"]

    return X, y


def split_data(X, y):

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


def get_models():

    return {
        "Logistic Regression":
            LogisticRegression(max_iter=1000),

        "Decision Tree":
            DecisionTreeClassifier(
                random_state=42
            ),

        "Random Forest":
            RandomForestClassifier(
                random_state=42
            ),

        "SVM":
            SVC(
                probability=True,
                random_state=42
            ),

        "KNN":
            KNeighborsClassifier()
    }


def train_models(X_train, X_test, y_train, y_test):

    models = get_models()

    results = []

    for name, model in models.items():

        model.fit(
            X_train,
            y_train
        )

        y_pred = model.predict(X_test)

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(
                y_test,
                y_pred
            ),
            "Precision": precision_score(
                y_test,
                y_pred
            ),
            "Recall": recall_score(
                y_test,
                y_pred
            ),
            "F1 Score": f1_score(
                y_test,
                y_pred
            )
        })

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="F1 Score",
        ascending=False
    )

    return models, results_df


if __name__ == "__main__":

    X, y = load_dataset()

    X_train, X_test, y_train, y_test = split_data(
        X,
        y
    )

    models, results = train_models(
        X_train,
        X_test,
        y_train,
        y_test
    )

    print("\nModel Comparison:\n")
    print(results)

    print("\nBest Model:\n")
    print(results.iloc[0])
