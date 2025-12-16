import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math

from my_utils import print_header


def print_default_info(df: pd.DataFrame):
    """Выводит базовую информацию о датасете (структура, примеры строк, пропуски)."""

    name_to_action: dict[str, tuple[callable, list]] = {
        "Первые строки датасета": lambda df: df.head(10),
        "Последние строки датасета": lambda df: df.tail(10),
        "Случайные строки": lambda df: df.sample(10),
        "Общая информация о данных": lambda df: df.info(),
        "Статистическое описание признаков": lambda df: df.describe().T,
        "Количество NULL-значений": lambda df: df.isnull().sum(),
        "Количество пропущенных значений (NaN)": lambda df: df.isna().sum(),
    }

    for name, action in name_to_action.items():
        print_header(name)
        print(action(df))


def perform_first_analyze(df: pd.DataFrame):
    """Выполняет первичный анализ данных и сохраняет основные визуализации."""

    print_header("Распределение целевого признака smoking")
    print(df["smoking"].value_counts(normalize=True))

    sns.countplot(x="smoking", data=df)
    plt.title("Распределение курящих и некурящих")
    plt.savefig("plots/smoking.png")
    plt.close()

    num_features = (
        df.select_dtypes(include=["int64", "float64"])
        .drop(columns=["id", "smoking"])
        .columns.tolist()
    )

    num_plots = len(num_features)
    n_cols = math.ceil(math.sqrt(num_plots))
    n_rows = math.ceil(num_plots / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(num_features):
        sns.boxplot(x="smoking", y=col, data=df, ax=axes[i])
        axes[i].set_title(col)

    # Удаляем пустые подграфики
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig("plots/boxplots_smoking.png", dpi=300)
    plt.close()

    print_header("Корреляционная матрица признаков")

    corr = df.drop(columns=["id"]).corr()
    print(corr)

    plt.figure(figsize=(14, 10))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Корреляционная матрица")
    plt.savefig("plots/correlation_matrix.png")
    plt.close()
