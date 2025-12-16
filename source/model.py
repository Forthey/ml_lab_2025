import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from my_types import Samples, VectorPair
from my_utils import print_header


class Model:
    """Класс для подготовки данных, обучения и оценки модели машинного обучения."""

    def __init__(self, df: pd.DataFrame, test_size=0.2, random_state=30):
        """Инициализирует модель и подготавливает обучающую и тестовую выборки."""
        self.df = df
        self.random_state = random_state
        self.samples = self._prepare_data(test_size)

    def _build_pipeline(self) -> Pipeline:
        """Создаёт sklearn Pipeline с нормализацией и Random Forest классификатором."""
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    RandomForestClassifier(
                        random_state=self.random_state,
                        class_weight="balanced",
                        criterion="gini",
                        n_jobs=-1,
                    ),
                ),
            ]
        )

    def _filter_features(self, x: pd.DataFrame, y: pd.DataFrame):
        """Исключает заранее заданные признаки из набора данных."""
        final_features = x.columns.tolist()
        columns = [
            "eyesight(left)",
            "eyesight(right)",
            "hearing(left)",
            "hearing(right)",
        ]
        for column in columns:
            final_features.remove(column)

        print("Используемые признаки:", final_features)
        return final_features

    def _filter_outliers(self, x: pd.DataFrame):
        """Формирует маску строк без выбросов по IQR-методу."""
        Q1 = x.quantile(0.25)
        Q3 = x.quantile(0.75)
        IQR = Q3 - Q1

        mask = ~((x < (Q1 - 1.5 * IQR)) | (x > (Q3 + 1.5 * IQR))).any(axis=1)

        return mask

    def _prepare_data(self, test_size) -> Samples:
        """Очищает данные, выполняет разбиение и масштабирование признаков."""
        self.df.dropna(inplace=True)
        self.df.dropna(axis=1, inplace=True)

        x = self.df.drop(columns=["id", "smoking"])
        y = self.df["smoking"]

        x = self.df[self._filter_features(x, y)]

        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=y,
        )

        new_mask = self._filter_outliers(x_train)
        x_train = x_train[new_mask]
        y_train = y_train[new_mask]

        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)

        return Samples(
            test=VectorPair(x=x_test_scaled, y=y_test),
            train=VectorPair(x=x_train_scaled, y=y_train),
        )

    def train_model_basic(self):
        """Обучает базовую модель без подбора гиперпараметров."""
        print_header("Random Forest (базовая модель)")

        pipeline = self._build_pipeline()
        pipeline.fit(self.samples.train.x, self.samples.train.y)

        y_pred = pipeline.predict(self.samples.test.x)
        y_proba = pipeline.predict_proba(self.samples.test.x)[:, 1]

        print(classification_report(self.samples.test.y, y_pred))
        print("ROC-AUC:", roc_auc_score(self.samples.test.y, y_proba))

        self.basic_model = pipeline

    def train_model_grid_search(self):
        """Обучает модель с подбором гиперпараметров с помощью GridSearchCV."""
        print_header("Random Forest (подбор гиперпараметров)")

        pipeline = self._build_pipeline()

        param_dist = {
            "model__n_estimators": [400, 800, 1200],
            "model__max_depth": [None, 10, 30],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 5],
            "model__max_features": ["sqrt", "log2"],
        }

        grid = RandomizedSearchCV(
            pipeline,
            param_distributions=param_dist,
            n_iter=20,
            cv=3,
            scoring="roc_auc",
            n_jobs=-1,
            verbose=2,
        )

        grid.fit(self.samples.train.x, self.samples.train.y)

        print("Лучшие параметры модели:", grid.best_params_)

        best_model = grid.best_estimator_

        y_pred = best_model.predict(self.samples.test.x)
        y_proba = best_model.predict_proba(self.samples.test.x)[:, 1]

        print(classification_report(self.samples.test.y, y_pred))
        print("ROC-AUC:", roc_auc_score(self.samples.test.y, y_proba))

        self.best_model = best_model
