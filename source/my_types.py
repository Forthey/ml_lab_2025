from dataclasses import dataclass


@dataclass
class VectorPair:
    """Хранит пару признаков и целевой переменной (X и y)."""

    x: list
    y: list


@dataclass
class Samples:
    """Содержит обучающую и тестовую выборки."""

    test: VectorPair
    train: VectorPair
