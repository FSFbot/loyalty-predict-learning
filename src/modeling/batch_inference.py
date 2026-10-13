import argparse
import json
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd

from src.modeling.modeling_dataset import (
    IDENTIFIER_COLUMNS,
)
from src.modeling.track_experiment import TRACKING_DIR


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gera previsões a partir de um CSV."
    )

    parser.add_argument("--run-id", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)

    return parser.parse_args()


def load_saved_model(run_id: str):
    mlflow.set_tracking_uri(TRACKING_DIR.as_uri())

    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)

    outputs = run.outputs.model_outputs

    if len(outputs) != 1:
        raise ValueError(
            "A execução deve conter exatamente um modelo."
        )

    model_id = outputs[0].model_id

    model = mlflow.sklearn.load_model(
        f"models:/{model_id}"
    )

    config_path = client.download_artifacts(
        run_id,
        "decision_config.json",
    )

    config = json.loads(
        Path(config_path).read_text(encoding="utf-8")
    )

    return model, config


def prepare_features(
    dataset: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    if dataset.empty:
        raise ValueError("O CSV de entrada está vazio.")

    required_columns = [
        *IDENTIFIER_COLUMNS,
        *feature_columns,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in dataset.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Colunas obrigatórias ausentes: {missing_columns}"
        )

    identifiers = dataset.loc[:, list(IDENTIFIER_COLUMNS)]

    if identifiers.isna().any().any():
        raise ValueError("Existem identificadores ausentes.")

    if identifiers.duplicated().any():
        raise ValueError("Existem observações cliente-mês duplicadas.")

    features = (
        dataset.loc[:, feature_columns]
        .apply(pd.to_numeric, errors="raise")
        .astype("float64")
    )

    if features.isin([float("inf"), float("-inf")]).any().any():
        raise ValueError("Existem valores infinitos nas características.")

    return features


def main() -> None:
    args = parse_arguments()

    if args.input.resolve() == args.output.resolve():
        raise ValueError("Entrada e saída devem ser arquivos diferentes.")

    model, config = load_saved_model(args.run_id)

    threshold = float(config["decision_threshold"])

    if not 0 <= threshold <= 1:
        raise ValueError("O limiar deve estar entre 0 e 1.")

    dataset = pd.read_csv(
        args.input,
        dtype={
            column: "string"
            for column in IDENTIFIER_COLUMNS
        },
    )

    features = prepare_features(
        dataset,
        config["feature_columns"],
    )

    positive_index = list(model.classes_).index(
        config["positive_class"]
    )

    probabilities = model.predict_proba(features)[
        :,
        positive_index,
    ]

    predictions = (probabilities >= threshold).astype(int)

    result = dataset.loc[:, list(IDENTIFIER_COLUMNS)].copy()

    result["probability_active_next_month"] = probabilities
    result["predicted_active_next_month"] = predictions
    result["decision_threshold"] = threshold
    result["run_id"] = args.run_id

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)

    print("Inferência em lote: OK")
    print(f"Observações: {len(result):,}")
    print(f"Limiar: {threshold:.2f}")
    print(f"Arquivo gerado: {args.output.resolve()}")


if __name__ == "__main__":
    main()