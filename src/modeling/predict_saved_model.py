import argparse
import json
from pathlib import Path

import mlflow
import mlflow.sklearn

from src.modeling.modeling_dataset import (
    IDENTIFIER_COLUMNS,
    build_modeling_dataset,
    load_feature_rows,
    validate_modeling_dataset,
)
from src.modeling.temporal_split import (
    split_modeling_dataset,
    validate_temporal_split,
)
from src.modeling.track_experiment import (
    TRACKING_DIR,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Carrega um modelo do MLflow "
            "e demonstra suas previsões."
        )
    )

    parser.add_argument(
        "--run-id",
        required=True,
        help="ID da execução registrada no MLflow.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    mlflow.set_tracking_uri(
        TRACKING_DIR.as_uri()
    )

    client = mlflow.tracking.MlflowClient()
    run = client.get_run(args.run_id)

    model_outputs = run.outputs.model_outputs

    if len(model_outputs) != 1:
        raise ValueError(
            "A execução deve conter exatamente "
            "um modelo registrado."
        )

    model_id = model_outputs[0].model_id

    model = mlflow.sklearn.load_model(
        f"models:/{model_id}"
    )

    config_path = client.download_artifacts(
        args.run_id,
        "decision_config.json",
    )

    config = json.loads(
        Path(config_path).read_text(
            encoding="utf-8"
        )
    )

    feature_columns = config["feature_columns"]
    target_column = config["target_column"]
    threshold = float(
        config["decision_threshold"]
    )
    positive_class = config["positive_class"]

    if not 0 <= threshold <= 1:
        raise ValueError(
            "O limiar salvo deve estar entre 0 e 1."
        )

    feature_rows = load_feature_rows()
    dataset = build_modeling_dataset(feature_rows)

    validate_modeling_dataset(
        dataset,
        expected_row_count=len(feature_rows),
    )

    split = split_modeling_dataset(dataset)
    validate_temporal_split(dataset, split)

    sample = split.test.head(5).copy()

    features = sample.loc[
        :,
        feature_columns,
    ].astype("float64")

    positive_class_index = list(
        model.classes_
    ).index(positive_class)

    probabilities = model.predict_proba(
        features
    )[:, positive_class_index]

    predictions = (
        probabilities >= threshold
    ).astype(int)

    result = sample.loc[
        :,
        [
            *IDENTIFIER_COLUMNS,
            target_column,
        ],
    ].copy()

    result["probability_active_next_month"] = (
        probabilities
    )
    result["predicted_active_next_month"] = (
        predictions
    )

    print(f"Run ID: {args.run_id}")
    print(f"Modelo carregado: {model_id}")
    print(f"Limiar salvo: {threshold:.2f}")
    print("\nExemplo de inferência:")

    print(
        result.to_string(
            index=False,
            formatters={
                "probability_active_next_month": (
                    lambda value: f"{value:.2%}"
                ),
            },
        )
    )


if __name__ == "__main__":
    main()