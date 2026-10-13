import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
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


THRESHOLDS = (
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
)


def build_threshold_analysis(
    y_true: pd.Series,
    y_score: pd.Series,
) -> pd.DataFrame:
    records = []

    for threshold in THRESHOLDS:
        y_pred = (
            y_score >= threshold
        ).astype(int)

        matrix = confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1],
        )

        tn, fp, fn, tp = matrix.ravel()

        records.append(
            {
                "threshold": threshold,
                "precision": precision_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                ),
                "recall": recall_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                ),
                "f1": f1_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                ),
                "predicted_positive_rate": float(
                    y_pred.mean()
                ),
                "true_negative": int(tn),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_positive": int(tp),
            }
        )

    return pd.DataFrame.from_records(records)


def print_threshold_analysis(
    analysis: pd.DataFrame,
) -> None:
    formatters = {
        "threshold": lambda value: f"{value:.2f}",
        "precision": lambda value: f"{value:.2%}",
        "recall": lambda value: f"{value:.2%}",
        "f1": lambda value: f"{value:.2%}",
        "predicted_positive_rate": (
            lambda value: f"{value:.2%}"
        ),
    }

    print("\nAnálise de limiares — validação")
    print(
        analysis.to_string(
            index=False,
            formatters=formatters,
        )
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

    split = split_modeling_dataset(dataset)

    validate_temporal_split(
        dataset,
        split,
    )

    model = train_logistic_model(split.train)

    x_validation = split.validation.loc[
        :,
        FEATURE_COLUMNS,
    ]

    y_true = split.validation[TARGET_COLUMN]

    positive_class_index = list(
        model.classes_
    ).index(1)

    probabilities = model.predict_proba(
        x_validation
    )[:, positive_class_index]

    y_score = pd.Series(
        probabilities,
        index=y_true.index,
        name="probability_active_next_month",
    )

    analysis = build_threshold_analysis(
        y_true,
        y_score,
    )

    print_threshold_analysis(analysis)


if __name__ == "__main__":
    main()