from dataclasses import dataclass

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.modeling.modeling_dataset import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    build_modeling_dataset,
    load_feature_rows,
    validate_modeling_dataset,
)
from src.modeling.temporal_split import (
    TemporalSplit,
    split_modeling_dataset,
    validate_temporal_split,
)


@dataclass(frozen=True)
class ClassificationMetrics:
    accuracy: float
    balanced_accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    average_precision: float
    predicted_positive_rate: float
    true_negative: int
    false_positive: int
    false_negative: int
    true_positive: int


def train_dummy_baseline(
    train_dataset: pd.DataFrame,
) -> DummyClassifier:
    x_train = train_dataset.loc[
        :,
        FEATURE_COLUMNS,
    ]
    y_train = train_dataset[TARGET_COLUMN]

    model = DummyClassifier(
        strategy="most_frequent"
    )

    model.fit(
        x_train,
        y_train,
    )

    return model


def evaluate_classifier(
    model: DummyClassifier,
    dataset: pd.DataFrame,
) -> ClassificationMetrics:
    x = dataset.loc[
        :,
        FEATURE_COLUMNS,
    ]
    y_true = dataset[TARGET_COLUMN]

    y_pred = model.predict(x)

    positive_class_index = list(
        model.classes_
    ).index(1)

    y_score = model.predict_proba(x)[
        :,
        positive_class_index,
    ]

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    true_negative, false_positive, false_negative, true_positive = (
        matrix.ravel()
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    predicted_positive_rate = float(
        y_pred.mean()
    )

    return ClassificationMetrics(
        accuracy=float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),
        balanced_accuracy=float(
            balanced_accuracy_score(
                y_true,
                y_pred,
            )
        ),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        roc_auc=float(
            roc_auc_score(
                y_true,
                y_score,
            )
        ),
        average_precision=float(
            average_precision_score(
                y_true,
                y_score,
            )
        ),
        predicted_positive_rate=(
            predicted_positive_rate
        ),
        true_negative=int(true_negative),
        false_positive=int(false_positive),
        false_negative=int(false_negative),
        true_positive=int(true_positive),
    )


def validate_classification_metrics(
    metrics: ClassificationMetrics,
    expected_row_count: int,
) -> None:
    bounded_metrics = {
        "accuracy": metrics.accuracy,
        "balanced_accuracy": metrics.balanced_accuracy,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1": metrics.f1,
        "roc_auc": metrics.roc_auc,
        "average_precision": metrics.average_precision,
        "predicted_positive_rate": (
            metrics.predicted_positive_rate
        ),
    }

    for metric_name, metric_value in (
        bounded_metrics.items()
    ):
        if not 0 <= metric_value <= 1:
            raise ValueError(
                f"A métrica {metric_name} está fora "
                "do intervalo entre 0 e 1."
            )

    confusion_total = (
        metrics.true_negative
        + metrics.false_positive
        + metrics.false_negative
        + metrics.true_positive
    )

    if confusion_total != expected_row_count:
        raise ValueError(
            "A matriz de confusão não corresponde à quantidade "
            "de linhas avaliadas."
        )


def print_classification_metrics(
    dataset_name: str,
    metrics: ClassificationMetrics,
) -> None:
    print(f"\n{dataset_name}")
    print(f"  Acurácia: {metrics.accuracy:.2%}")
    print(
        "  Acurácia balanceada: "
        f"{metrics.balanced_accuracy:.2%}"
    )
    print(f"  Precisão: {metrics.precision:.2%}")
    print(f"  Recall: {metrics.recall:.2%}")
    print(f"  F1: {metrics.f1:.2%}")
    print(f"  ROC AUC: {metrics.roc_auc:.2%}")
    print(
        "  Average precision: "
        f"{metrics.average_precision:.2%}"
    )
    print(
        "  Taxa prevista de positivos: "
        f"{metrics.predicted_positive_rate:.2%}"
    )

    print("  Matriz de confusão:")
    print(
        "    Verdadeiros negativos: "
        f"{metrics.true_negative:,}"
    )
    print(
        "    Falsos positivos: "
        f"{metrics.false_positive:,}"
    )
    print(
        "    Falsos negativos: "
        f"{metrics.false_negative:,}"
    )
    print(
        "    Verdadeiros positivos: "
        f"{metrics.true_positive:,}"
    )


def main() -> None:
    feature_rows = load_feature_rows()

    dataset = build_modeling_dataset(
        feature_rows
    )
    validate_modeling_dataset(
        dataset,
        expected_row_count=len(feature_rows),
    )

    split: TemporalSplit = split_modeling_dataset(
        dataset
    )
    validate_temporal_split(
        dataset,
        split,
    )

    model = train_dummy_baseline(
        split.train
    )

    validation_metrics = evaluate_classifier(
        model,
        split.validation,
    )

    validate_classification_metrics(
        validation_metrics,
        expected_row_count=len(split.validation),
    )

    print("Validação do baseline ingênuo: OK")
    majority_class_index = (
    model.class_prior_.argmax()
)

majority_class = model.classes_[
    majority_class_index
]

print(
    "Estratégia: sempre prever a classe "
    f"{int(majority_class)}"
)

    print_classification_metrics(
        "Validação",
        validation_metrics,
    )


if __name__ == "__main__":
    main()