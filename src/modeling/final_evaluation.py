import pandas as pd

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
from sklearn.pipeline import Pipeline

from src.modeling.baseline_model import (
    ClassificationMetrics,
    print_classification_metrics,
    validate_classification_metrics,
)
from src.modeling.logistic_model import (
    train_logistic_model,
)
from src.modeling.modeling_dataset import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    build_modeling_dataset,
    load_feature_rows,
    validate_modeling_dataset,
)
from src.modeling.temporal_split import (
    split_modeling_dataset,
    validate_temporal_split,
)


DECISION_THRESHOLD = 0.30


def evaluate_final_model(
    model: Pipeline,
    dataset: pd.DataFrame,
    threshold: float,
) -> ClassificationMetrics:
    if not 0 <= threshold <= 1:
        raise ValueError(
            "O limiar deve estar entre 0 e 1."
        )

    x = dataset.loc[:, FEATURE_COLUMNS]
    y_true = dataset[TARGET_COLUMN]

    positive_class_index = list(
        model.classes_
    ).index(1)

    y_score = model.predict_proba(x)[
        :,
        positive_class_index,
    ]

    y_pred = (
        y_score >= threshold
    ).astype(int)

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    tn, fp, fn, tp = matrix.ravel()

    return ClassificationMetrics(
        accuracy=float(
            accuracy_score(y_true, y_pred)
        ),
        balanced_accuracy=float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        precision=float(
            precision_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        recall=float(
            recall_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        f1=float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        roc_auc=float(
            roc_auc_score(y_true, y_score)
        ),
        average_precision=float(
            average_precision_score(
                y_true,
                y_score,
            )
        ),
        predicted_positive_rate=float(
            y_pred.mean()
        ),
        true_negative=int(tn),
        false_positive=int(fp),
        false_negative=int(fn),
        true_positive=int(tp),
    )


def main() -> None:
    feature_rows = load_feature_rows()

    dataset = build_modeling_dataset(feature_rows)

    validate_modeling_dataset(
        dataset,
        expected_row_count=len(feature_rows),
    )

    split = split_modeling_dataset(dataset)
    validate_temporal_split(dataset, split)

    model = train_logistic_model(split.train)

    metrics = evaluate_final_model(
        model,
        split.test,
        threshold=DECISION_THRESHOLD,
    )

    validate_classification_metrics(
        metrics,
        expected_row_count=len(split.test),
    )

    print("Avaliação final da regressão logística")
    print(
        f"Limiar fixado na validação: "
        f"{DECISION_THRESHOLD:.2f}"
    )
    print(
        "Período de teste: "
        f"{split.test['reference_month'].min():%Y-%m}"
        " até "
        f"{split.test['reference_month'].max():%Y-%m}"
    )
    print(
        f"Observações cliente-mês: {len(split.test):,}"
    )
    print(
        "Taxa real de atividade no teste: "
        f"{split.test[TARGET_COLUMN].mean():.2%}"
    )

    print_classification_metrics(
        "Teste",
        metrics,
    )


if __name__ == "__main__":
    main()