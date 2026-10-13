from dataclasses import asdict

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

from src.engineering.config import PROJECT_ROOT
from src.modeling.baseline_model import (
    validate_classification_metrics,
)
from src.modeling.final_evaluation import (
    DECISION_THRESHOLD,
    evaluate_final_model,
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


EXPERIMENT_NAME = "loyalty-predict-learning"
TRACKING_DIR = PROJECT_ROOT / "mlruns"


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

    validation_metrics = evaluate_final_model(
        model,
        split.validation,
        threshold=DECISION_THRESHOLD,
    )
    test_metrics = evaluate_final_model(
        model,
        split.test,
        threshold=DECISION_THRESHOLD,
    )

    validate_classification_metrics(
        validation_metrics,
        expected_row_count=len(split.validation),
    )
    validate_classification_metrics(
        test_metrics,
        expected_row_count=len(split.test),
    )

    TRACKING_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    mlflow.set_tracking_uri(
        TRACKING_DIR.as_uri()
    )
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(
        run_name="logistic-regression-threshold-030"
    ) as run:
        classifier = model.named_steps["classifier"]
        imputer = model.named_steps["imputer"]

        mlflow.log_params(
            {
                "model_type": "logistic_regression",
                "decision_threshold": DECISION_THRESHOLD,
                "threshold_selection": (
                    "highest_validation_f1_in_tested_grid"
                ),
                "max_iter": classifier.max_iter,
                "C": classifier.C,
                "solver": classifier.solver,
                "imputer_strategy": imputer.strategy,
                "add_indicator": imputer.add_indicator,
                "scaler": "StandardScaler",
                "feature_count": len(FEATURE_COLUMNS),
                "target": TARGET_COLUMN,
            }
        )

        for split_name, split_dataset in (
            {
                "train": split.train,
                "validation": split.validation,
                "test": split.test,
            }.items()
        ):
            mlflow.log_params(
                {
                    f"{split_name}_rows": len(split_dataset),
                    f"{split_name}_start_month": (
                        split_dataset["reference_month"]
                        .min()
                        .strftime("%Y-%m")
                    ),
                    f"{split_name}_end_month": (
                        split_dataset["reference_month"]
                        .max()
                        .strftime("%Y-%m")
                    ),
                }
            )

        for split_name, metrics in (
            {
                "validation": validation_metrics,
                "test": test_metrics,
            }.items()
        ):
            mlflow.log_metrics(
                {
                    f"{split_name}_{metric_name}": float(value)
                    for metric_name, value in (
                        asdict(metrics).items()
                    )
                }
            )

        mlflow.log_dict(
            {
                "decision_threshold": DECISION_THRESHOLD,
                "positive_class": 1,
                "feature_columns": list(FEATURE_COLUMNS),
                "target_column": TARGET_COLUMN,
            },
            "decision_config.json",
        )

        input_example = (
            split.train.loc[:, FEATURE_COLUMNS]
            .head(5)
            .astype("float64")
        )

        signature = infer_signature(
            input_example,
            model.predict(input_example).astype("float64"),
        )

        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="logistic_pipeline",
            input_example=input_example,
            signature=signature,
        )
        print("Registro do experimento: OK")
        print(f"Run ID: {run.info.run_id}")
        print(f"Modelo: {model_info.model_uri}")
        print(f"Registros locais: {TRACKING_DIR}")


if __name__ == "__main__":
    main()