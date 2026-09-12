import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def class_distribution(df):

    plt.figure(figsize=(6, 4))

    df["TYPE"].value_counts().plot(
        kind="bar"
    )

    plt.title("Cancer Type Distribution")
    plt.xlabel("Cancer Type")
    plt.ylabel("Count")

    plt.tight_layout()
    plt.show()


def biomarker_summary(df):

    print("\nMean CA125 and HE4 by Cancer Type:\n")

    print(
        df.groupby("TYPE")[["CA125", "HE4"]]
        .mean()
    )


def plot_ca125(df):

    plt.figure(figsize=(7, 5))

    sns.boxplot(
        x="TYPE",
        y="CA125",
        data=df
    )

    plt.title("CA125 Distribution by Cancer Type")
    plt.xlabel("Cancer Type")
    plt.ylabel("CA125")

    plt.tight_layout()
    plt.show()


def plot_he4(df):

    plt.figure(figsize=(7, 5))

    sns.boxplot(
        x="TYPE",
        y="HE4",
        data=df
    )

    plt.title("HE4 Distribution by Cancer Type")
    plt.xlabel("Cancer Type")
    plt.ylabel("HE4")

    plt.tight_layout()
    plt.show()


def correlation_analysis(df):

    columns = ["CA125", "HE4", "TYPE"]

    correlation = df[columns].corr()

    print("\nCorrelation Matrix:\n")
    print(correlation)

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        correlation,
        annot=True,
        cmap="coolwarm"
    )

    plt.title(
        "Correlation Between Biomarkers and Cancer Type"
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    df = pd.read_csv(
        "data/processed/data.csv"
    )

    class_distribution(df)
    biomarker_summary(df)
    plot_ca125(df)
    plot_he4(df)
    correlation_analysis(df)
