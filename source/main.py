from dataclasses import dataclass
import argparse

from eda import print_default_info, perform_first_analyze
from my_utils import get_data
from model import Model


@dataclass
class CommandLineArgs:
    data_path: str


def main():
    parser = argparse.ArgumentParser(
        description="Программа для обучения по показателям здоровья предсказывать, курит ли пациент"
    )

    parser.add_argument(
        "-d",
        "--data_path",
        help="Путь к .csv файлу с датасетом",
        type=str,
        default="data/data.csv",
    )
    args: CommandLineArgs = parser.parse_args()

    data = get_data(args.data_path)
    if data is None or data.empty:
        print(f"Could not read file {args.data_path}")
        exit(1)

    print_default_info(data)
    perform_first_analyze(data)

    model = Model(data)
    model.train_model_basic()
    model.train_model_grid_search()


if __name__ == "__main__":
    main()
