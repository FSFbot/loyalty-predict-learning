import argparse
from datetime import datetime, timezone

from src.engineering.config import DATA_DIR
from src.modeling.batch_inference import (
    load_saved_model,
    prepare_features,
)
from src.modeling.export_inference_features import (
    build_inference_features,
)
from src.modeling.modeling_dataset import (
    IDENTIFIER_COLUMNS,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Executa o fluxo completo de inferência."
        )
    )

    parser.add_argument(
        "--run-id",
        required=True,
        help="ID da execução do modelo no MLflow.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    model, config = load_saved_model(args.run_id)

    threshold = float(config["decision_threshold"])

    if not 0 <= threshold <= 1:
        raise ValueError(
            "O limiar deve estar entre 0 e 1."
        )

    dataset = build_inference_features()

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

    predictions = (
        probabilities >= threshold
    ).astype(int)

    result = dataset.loc[
        :,
        list(IDENTIFIER_COLUMNS),
    ].copy()

    result["probability_active_next_month"] = probabilities
    result["predicted_active_next_month"] = predictions
    result["decision_threshold"] = threshold
    result["run_id"] = args.run_id

    execution_id = datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%S%fZ")

    output_dir = (
        DATA_DIR
        / "inference"
        / "runs"
        / execution_id
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    features_path = output_dir / "features.csv"
    predictions_path = output_dir / "predictions.csv"

    dataset.to_csv(features_path, index=False)
    result.to_csv(predictions_path, index=False)

    reference_month = (
        dataset["reference_month"].iloc[0]
    )

    positive_count = int(
        result["predicted_active_next_month"].sum()
    )

    print("\nFluxo completo de inferência: OK")
    print(f"Execução da inferência: {execution_id}")
    print(f"Run ID do modelo: {args.run_id}")
    print(
        f"Mês de referência: {reference_month:%Y-%m}"
    )
    print(f"Observações: {len(result):,}")
    print(f"Limiar: {threshold:.2f}")
    print(f"Previsões de atividade: {positive_count:,}")
    print(f"Características: {features_path}")
    print(f"Previsões: {predictions_path}")


if __name__ == "__main__":
    main()