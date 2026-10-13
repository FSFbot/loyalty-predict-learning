import sqlite3
from contextlib import closing
from dataclasses import asdict

import pandas as pd

from src.analytics import customer_month as cm
from src.analytics import education_month as em
from src.analytics import historical_features as hf
from src.analytics import recency_features as rf
from src.analytics import lifecycle_features as lf
from src.analytics import integrated_features as inf
from src.analytics import education_history_features as ehf

from src.analytics.modeling_table import build_modeling_table
from src.engineering.config import DATA_DIR, KAGGLE_DATASETS
from src.modeling.batch_inference import prepare_features
from src.modeling.modeling_dataset import (
    FEATURE_COLUMNS,
    IDENTIFIER_COLUMNS,
    NULLABLE_FEATURE_COLUMNS,
)


OUTPUT_PATH = (
    DATA_DIR / "inference" / "latest_features.csv"
)


def build_inference_features() -> pd.DataFrame:
    databases = {
        dataset.destination.name: (
            dataset.destination / "database.db"
        )
        for dataset in KAGGLE_DATASETS
    }

    for database_path in databases.values():
        if not database_path.is_file():
            raise FileNotFoundError(database_path)

    with closing(
        sqlite3.connect(
            databases["loyalty"].resolve().as_uri()
            + "?mode=ro",
            uri=True,
        )
    ) as connection:
        customer_months = cm.load_customer_months(connection)
        active_months = rf.load_active_months(connection)

    with closing(
        sqlite3.connect(
            databases["education"].resolve().as_uri()
            + "?mode=ro",
            uri=True,
        )
    ) as connection:
        education_months = em.load_education_months(connection)

    cm.validate_customer_months(customer_months)
    em.validate_education_months(education_months)

    modeling_rows = build_modeling_table(
        customer_months,
        targets=[],
        require_target=False,
    )

    historical_rows = hf.build_historical_features(
        modeling_rows
    )
    hf.validate_historical_features(
        historical_rows,
        modeling_rows,
        allow_unknown_target=True,
    )

    recency_rows = rf.build_recency_features(
        historical_rows,
        active_months,
    )
    rf.validate_recency_features(
        recency_rows,
        historical_rows,
    )

    lifecycle_rows = lf.build_lifecycle_features(
        recency_rows
    )
    lf.validate_lifecycle_features(
        lifecycle_rows,
        recency_rows,
    )

    integrated_rows = inf.build_integrated_features(
        lifecycle_rows,
        education_months,
    )
    inf.validate_integrated_features(
        integrated_rows,
        lifecycle_rows,
    )

    feature_rows = ehf.build_education_history_features(
        integrated_rows,
        education_months,
    )
    ehf.validate_education_history_features(
        feature_rows,
        integrated_rows,
    )

    dataset = pd.DataFrame.from_records(
        [asdict(row) for row in feature_rows]
    )

    columns = [
        *IDENTIFIER_COLUMNS,
        *FEATURE_COLUMNS,
    ]
    dataset = dataset.loc[:, columns].copy()

    numeric_features = prepare_features(
        dataset,
        list(FEATURE_COLUMNS),
    )

    required_features = [
        column
        for column in FEATURE_COLUMNS
        if column not in NULLABLE_FEATURE_COLUMNS
    ]

    if numeric_features[required_features].isna().any().any():
        raise ValueError(
            "Ausências em características obrigatórias."
        )

    dataset[list(FEATURE_COLUMNS)] = numeric_features

    dataset["reference_month"] = pd.to_datetime(
        dataset["reference_month"],
        format="%Y-%m",
        errors="raise",
    )

    latest_month = dataset["reference_month"].max()

    latest = dataset.loc[
        dataset["reference_month"] == latest_month
    ].copy()

    latest = latest.sort_values(
        "customer_id"
    ).reset_index(drop=True)

    return latest


def main() -> None:
    dataset = build_inference_features()

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset.to_csv(OUTPUT_PATH, index=False)

    reference_month = dataset["reference_month"].iloc[0]
    prediction_month = reference_month + pd.DateOffset(months=1)

    print("Exportação das características: OK")
    print(f"Mês de referência: {reference_month:%Y-%m}")
    print(f"Mês a prever: {prediction_month:%Y-%m}")
    print(f"Observações: {len(dataset):,}")
    print(f"Características: {len(FEATURE_COLUMNS)}")
    print(f"Arquivo gerado: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()