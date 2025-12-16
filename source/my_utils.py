import pandas as pd


def print_header(text: str, align: int = 20):
    """Печатает форматированный заголовок для логов и вывода в консоль."""
    print(f"\n{'=' * align} {text} {'=' * align}\n")


def get_data(filename: str) -> pd.DataFrame | None:
    """Загружает CSV-файл в DataFrame и возвращает None, если файл не найден."""
    try:
        return pd.read_csv(filename)
    except FileNotFoundError:
        print(f"Файл не найден: {filename}")
        return None
