import sqlite3
from contextlib import closing
from dataclasses import asdict

import pandas as pd
from pandas.api.types import is_numeric_dtype

from src.analytics.customer_month import (
    load_customer_months,
    validate_customer_months,
)
from src.analytics.customer_month_target import (
    build_customer_month_targets,
    validate_customer_month_targets,
)
from src.analytics.education_history_features import (
    EducationHistoryFeatureRow,
    build_education_history_features,
    validate_education_history_features,
)
from src.analytics.education_month import (
    load_education_months,
    validate_education_months,
)
from src.analytics.historical_features import (
    build_historical_features,
    validate_historical_features,
)
from src.analytics.integrated_features import (
    build_integrated_features,
    validate_integrated_features,
)
from src.analytics.lifecycle_features import (
    build_lifecycle_features,
    validate_lifecycle_features,
)
from src.analytics.modeling_table import (
    build_modeling_table,
    validate_modeling_table,
)
from src.analytics.recency_features import (
    build_recency_features,
    load_active_months,
    validate_recency_features,
)
from src.engineering.config import KAGGLE_DATASETS


IDENTIFIER_COLUMNS = (
    "customer_id",
    "reference_month",
)

FEATURE_COLUMNS = (
    "transaction_count",
    "positive_transaction_count",
    "negative_transaction_count",
    "zero_point_transaction_count",
    "points_added",
    "points_removed",
    "points_moved",
    "net_points",
    "observed_months_last_3",
    "active_months_last_3",
    "transaction_count_last_3_months",
    "net_points_last_3_months",
    "months_since_last_activity",
    "complete_months_observed",
    "active_months_to_date",
    "transaction_count_to_date",
    "net_points_to_date",
    "activity_rate_to_date",
    "completed_episode_count",
    "distinct_course_count",
    "skill_record_count",
    "distinct_skill_count",
    "reward_count",
    "education_event_count",
    "has_education_activity",
    "education_active_months_last_3",
    "education_event_count_last_3_months",
    "months_since_last_education_activity",
    "education_active_months_to_date",
    "education_event_count_to_date",
)

TARGET_COLUMN = "target_active_next_month"

NULLABLE_FEATURE_COLUMNS = (
    "months_since_last_activity",
    "months_since_last_education_activity",
)


def load_feature_rows() -> list[EducationHistoryFeatureRow]:
    datasets = {
        dataset.destination.name: dataset
        for dataset in KAGGLE_DATASETS
    }

    loyalty_database = (
        datasets["loyalty"].destination
        / "database.db"
    )
    education_database = (
        datasets["education"].destination
        / "database.db"
    )

    with closing(
        sqlite3.connect(loyalty_database)
    ) as loyalty_connection:
        customer_months = load_customer_months(
            loyalty_connection
        )
        active_months_by_customer = load_active_months(
            loyalty_connection
        )

    with closing(
        sqlite3.connect(education_database)
    ) as education_connection:
        education_months = load_education_months(
            education_connection
        )

    validate_customer_months(customer_months)
    validate_education_months(education_months)

    targets = build_customer_month_targets(
        customer_months
    )
    validate_customer_month_targets(targets)

    modeling_rows = build_modeling_table(
        customer_months,
        targets,
    )
    validate_modeling_table(
        modeling_rows,
        targets,
    )

    historical_rows = build_historical_features(
        modeling_rows
    )
    validate_historical_features(
        historical_rows,
        modeling_rows,
    )

    recency_rows = build_recency_features(
        historical_rows,
        active_months_by_customer,
    )
    validate_recency_features(
        recency_rows,
        historical_rows,
    )

    lifecycle_rows = build_lifecycle_features(
        recency_rows
    )
    validate_lifecycle_features(
        lifecycle_rows,
        recency_rows,
    )

    integrated_rows = build_integrated_features(
        lifecycle_rows,
        education_months,
    )
    validate_integrated_features(
        integrated_rows,
        lifecycle_rows,
    )

    feature_rows = build_education_history_features(
        integrated_rows,
        education_months,
    )
    validate_education_history_features(
        feature_rows,
        integrated_rows,
    )

    return feature_rows


def build_modeling_dataset(
    feature_rows: list[EducationHistoryFeatureRow],
) -> pd.DataFrame:
    records = [
        asdict(row)
        for row in feature_rows
    ]

    dataset = pd.DataFrame.from_records(records)

    dataset["reference_month"] = pd.to_datetime(
        dataset["reference_month"],
        format="%Y-%m",
        errors="raise",
    )

    dataset["customer_id"] = (
        dataset["customer_id"].astype("string")
    )

    for column in NULLABLE_FEATURE_COLUMNS:
        dataset[column] = dataset[column].astype(
            "Int64"
        )

    dataset[TARGET_COLUMN] = dataset[
        TARGET_COLUMN
    ].astype("int8")

    ordered_columns = (
        *IDENTIFIER_COLUMNS,
        *FEATURE_COLUMNS,
        TARGET_COLUMN,
    )

    dataset = dataset.loc[:, ordered_columns]

    dataset = (
        dataset.sort_values(
            list(IDENTIFIER_COLUMNS)
        )
        .reset_index(drop=True)
    )

    return dataset


def validate_modeling_dataset(
    dataset: pd.DataFrame,
    expected_row_count: int,
) -> None:
    if dataset.empty:
        raise ValueError(
            "O DataFrame de modelagem está vazio."
        )

    if len(dataset) != expected_row_count:
        raise ValueError(
            "A quantidade de linhas mudou durante a conversão "
            "para DataFrame."
        )

    expected_columns = [
        *IDENTIFIER_COLUMNS,
        *FEATURE_COLUMNS,
        TARGET_COLUMN,
    ]

    if dataset.columns.tolist() != expected_columns:
        raise ValueError(
            "As colunas do DataFrame não correspondem "
            "ao esquema esperado."
        )

    duplicated_keys = dataset.duplicated(
        subset=list(IDENTIFIER_COLUMNS),
        keep=False,
    )

    if duplicated_keys.any():
        raise ValueError(
            "O DataFrame possui combinações cliente-mês duplicadas."
        )

    if dataset["customer_id"].isna().any():
        raise ValueError(
            "Existem identificadores de cliente ausentes."
        )

    if dataset["reference_month"].isna().any():
        raise ValueError(
            "Existem meses de referência ausentes."
        )

    target_values = set(
        dataset[TARGET_COLUMN].unique()
    )

    if not target_values.issubset({0, 1}):
        raise ValueError(
            "O alvo contém valores diferentes de 0 e 1."
        )

    non_numeric_features = [
        column
        for column in FEATURE_COLUMNS
        if not is_numeric_dtype(dataset[column])
    ]

    if non_numeric_features:
        raise ValueError(
            "Existem características não numéricas: "
            f"{non_numeric_features}."
        )

    allowed_missing_columns = set(
        NULLABLE_FEATURE_COLUMNS
    )

    unexpected_missing_columns = [
        column
        for column in dataset.columns
        if column not in allowed_missing_columns
        and dataset[column].isna().any()
    ]

    if unexpected_missing_columns:
        raise ValueError(
            "Existem valores ausentes em colunas não permitidas: "
            f"{unexpected_missing_columns}."
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

    target_counts = (
        dataset[TARGET_COLUMN]
        .value_counts()
        .sort_index()
    )

    missing_recencies = dataset[
        list(NULLABLE_FEATURE_COLUMNS)
    ].isna().sum()

    memory_megabytes = (
        dataset.memory_usage(
            deep=True
        ).sum()
        / 1024**2
    )

    print("Validação do DataFrame de modelagem: OK")
    print(f"Formato: {dataset.shape}")
    print(
        "Quantidade de características: "
        f"{len(FEATURE_COLUMNS)}"
    )
    print(
        "Período: "
        f"{dataset['reference_month'].min():%Y-%m} "
        "até "
        f"{dataset['reference_month'].max():%Y-%m}"
    )
    print(
        "Memória aproximada: "
        f"{memory_megabytes:.2f} MB"
    )

    print("\nDistribuição do alvo:")
    print(target_counts.to_string())

    print("\nValores ausentes permitidos:")
    print(missing_recencies.to_string())

    print("\nTipos dos identificadores e do alvo:")
    print(
        dataset[
            [
                *IDENTIFIER_COLUMNS,
                TARGET_COLUMN,
            ]
        ].dtypes.to_string()
    )


if __name__ == "__main__":
    main()