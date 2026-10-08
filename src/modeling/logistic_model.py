import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.modeling.baseline_model import (
    evaluate_classifier,
    print_classification_metrics,
    train_dummy_baseline,
    validate_classification_metrics,
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


def train_logistic_model(
    train_dataset: pd.DataFrame,
) -> Pipeline:
    x_train = train_dataset.loc[:, FEATURE_COLUMNS]
    y_train = train_dataset[TARGET_COLUMN]

    model = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                    keep_empty_features=True,
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                ),
            ),
        ],
    )

    model.fit(x_train, y_train)

    return model


def main() -> None:
    feature_rows = load_feature_rows()

    dataset = build_modeling_dataset(feature_rows)

    validate_modeling_dataset(
        dataset,
        expected_row_count=len(feature_rows),
    )

    split = split_modeling_dataset(dataset)
    validate_temporal_split(dataset, split)

    baseline = train_dummy_baseline(split.train)
    logistic_model = train_logistic_model(split.train)

    models = {
        "Baseline ingênuo — validação": baseline,
        "Regressão logística — validação": logistic_model,
    }

    for model_name, model in models.items():
        metrics = evaluate_classifier(
            model,
            split.validation,
        )

        validate_classification_metrics(
            metrics,
            expected_row_count=len(split.validation),
        )

        print_classification_metrics(
            model_name,
            metrics,
        )


if __name__ == "__main__":
    main()