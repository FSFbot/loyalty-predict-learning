import sqlite3
from collections import defaultdict
from contextlib import closing
from dataclasses import dataclass
from math import isclose

from src.analytics.customer_month import (
    load_customer_months,
    validate_customer_months,
)
from src.analytics.customer_month_target import (
    build_customer_month_targets,
    validate_customer_month_targets,
)
from src.analytics.historical_features import (
    build_historical_features,
    validate_historical_features,
)
from src.analytics.modeling_table import (
    build_modeling_table,
    validate_modeling_table,
)
from src.analytics.recency_features import (
    RecencyFeatureRow,
    build_recency_features,
    load_active_months,
    validate_recency_features,
)
from src.engineering.config import KAGGLE_DATASETS


@dataclass(frozen=True)
class LifecycleFeatureRow:
    customer_id: str
    reference_month: str
    transaction_count: int
    positive_transaction_count: int
    negative_transaction_count: int
    zero_point_transaction_count: int
    points_added: int
    points_removed: int
    points_moved: int
    net_points: int
    observed_months_last_3: int
    active_months_last_3: int
    transaction_count_last_3_months: int
    net_points_last_3_months: int
    months_since_last_activity: int | None
    complete_months_observed: int
    active_months_to_date: int
    transaction_count_to_date: int
    net_points_to_date: int
    activity_rate_to_date: float
    target_active_next_month: int


def build_lifecycle_features(
    recency_rows: list[RecencyFeatureRow],
) -> list[LifecycleFeatureRow]:
    rows_by_customer: defaultdict[
        str,
        list[RecencyFeatureRow],
    ] = defaultdict(list)

    for row in recency_rows:
        rows_by_customer[row.customer_id].append(row)

    lifecycle_rows: list[LifecycleFeatureRow] = []

    for customer_rows in rows_by_customer.values():
        customer_rows.sort(
            key=lambda row: row.reference_month
        )

        active_months_to_date = 0
        transaction_count_to_date = 0
        net_points_to_date = 0

        for complete_months_observed, row in enumerate(
            customer_rows,
            start=1,
        ):
            active_months_to_date += int(
                row.transaction_count > 0
            )
            transaction_count_to_date += (
                row.transaction_count
            )
            net_points_to_date += row.net_points

            activity_rate_to_date = (
                active_months_to_date
                / complete_months_observed
            )

            lifecycle_rows.append(
                LifecycleFeatureRow(
                    customer_id=row.customer_id,
                    reference_month=row.reference_month,
                    transaction_count=row.transaction_count,
                    positive_transaction_count=(
                        row.positive_transaction_count
                    ),
                    negative_transaction_count=(
                        row.negative_transaction_count
                    ),
                    zero_point_transaction_count=(
                        row.zero_point_transaction_count
                    ),
                    points_added=row.points_added,
                    points_removed=row.points_removed,
                    points_moved=row.points_moved,
                    net_points=row.net_points,
                    observed_months_last_3=(
                        row.observed_months_last_3
                    ),
                    active_months_last_3=(
                        row.active_months_last_3
                    ),
                    transaction_count_last_3_months=(
                        row.transaction_count_last_3_months
                    ),
                    net_points_last_3_months=(
                        row.net_points_last_3_months
                    ),
                    months_since_last_activity=(
                        row.months_since_last_activity
                    ),
                    complete_months_observed=(
                        complete_months_observed
                    ),
                    active_months_to_date=(
                        active_months_to_date
                    ),
                    transaction_count_to_date=(
                        transaction_count_to_date
                    ),
                    net_points_to_date=(
                        net_points_to_date
                    ),
                    activity_rate_to_date=(
                        activity_rate_to_date
                    ),
                    target_active_next_month=(
                        row.target_active_next_month
                    ),
                )
            )

    lifecycle_rows.sort(
        key=lambda row: (
            row.reference_month,
            row.customer_id,
        )
    )

    return lifecycle_rows


def validate_lifecycle_features(
    lifecycle_rows: list[LifecycleFeatureRow],
    recency_rows: list[RecencyFeatureRow],
) -> None:
    if not lifecycle_rows:
        raise ValueError(
            "A tabela de ciclo de vida está vazia."
        )

    if len(lifecycle_rows) != len(recency_rows):
        raise ValueError(
            "A quantidade de linhas mudou durante a construção "
            "das características de ciclo de vida."
        )

    expected_keys = {
        (
            row.customer_id,
            row.reference_month,
        )
        for row in recency_rows
    }

    observed_keys = {
        (
            row.customer_id,
            row.reference_month,
        )
        for row in lifecycle_rows
    }

    if observed_keys != expected_keys:
        raise ValueError(
            "As chaves foram alteradas durante a construção "
            "das características de ciclo de vida."
        )

    rows_by_customer: defaultdict[
        str,
        list[LifecycleFeatureRow],
    ] = defaultdict(list)

    for row in lifecycle_rows:
        rows_by_customer[row.customer_id].append(row)

    for customer_rows in rows_by_customer.values():
        customer_rows.sort(
            key=lambda row: row.reference_month
        )

        previous_row: LifecycleFeatureRow | None = None

        for row in customer_rows:
            if not (
                1
                <= row.active_months_to_date
                <= row.complete_months_observed
            ) and row.active_months_to_date != 0:
                raise ValueError(
                    "Quantidade acumulada de meses ativos inválida "
                    f"para {row.customer_id} em "
                    f"{row.reference_month}."
                )

            expected_activity_rate = (
                row.active_months_to_date
                / row.complete_months_observed
            )

            if not isclose(
                row.activity_rate_to_date,
                expected_activity_rate,
            ):
                raise ValueError(
                    "Taxa acumulada de atividade inválida para "
                    f"{row.customer_id} em "
                    f"{row.reference_month}."
                )

            if previous_row is None:
                expected_complete_months = 1
                expected_active_months = int(
                    row.transaction_count > 0
                )
                expected_transaction_count = (
                    row.transaction_count
                )
                expected_net_points = row.net_points
            else:
                expected_complete_months = (
                    previous_row.complete_months_observed + 1
                )
                expected_active_months = (
                    previous_row.active_months_to_date
                    + int(row.transaction_count > 0)
                )
                expected_transaction_count = (
                    previous_row.transaction_count_to_date
                    + row.transaction_count
                )
                expected_net_points = (
                    previous_row.net_points_to_date
                    + row.net_points
                )

            if (
                row.complete_months_observed
                != expected_complete_months
            ):
                raise ValueError(
                    "Sequência de meses observados inválida para "
                    f"{row.customer_id} em "
                    f"{row.reference_month}."
                )

            if (
                row.active_months_to_date
                != expected_active_months
            ):
                raise ValueError(
                    "Acúmulo de meses ativos inválido para "
                    f"{row.customer_id} em "
                    f"{row.reference_month}."
                )

            if (
                row.transaction_count_to_date
                != expected_transaction_count
            ):
                raise ValueError(
                    "Acúmulo de transações inválido para "
                    f"{row.customer_id} em "
                    f"{row.reference_month}."
                )

            if row.net_points_to_date != expected_net_points:
                raise ValueError(
                    "Acúmulo de pontos inválido para "
                    f"{row.customer_id} em "
                    f"{row.reference_month}."
                )

            previous_row = row


def main() -> None:
    datasets = {
        dataset.destination.name: dataset
        for dataset in KAGGLE_DATASETS
    }

    loyalty_database = (
        datasets["loyalty"].destination
        / "database.db"
    )

    with closing(
        sqlite3.connect(loyalty_database)
    ) as connection:
        customer_months = load_customer_months(
            connection
        )
        active_months_by_customer = load_active_months(
            connection
        )

    validate_customer_months(customer_months)

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

    maximum_observed_months = max(
        row.complete_months_observed
        for row in lifecycle_rows
    )

    print("Validação das características de ciclo de vida: OK")
    print(f"Total de linhas: {len(lifecycle_rows):,}")
    print(
        "Maior período completo observado: "
        f"{maximum_observed_months} meses"
    )

    print("\nPrimeiras 5 linhas com 3 meses observados:")

    examples = [
        row
        for row in lifecycle_rows
        if row.complete_months_observed == 3
    ]

    for row in examples[:5]:
        print(row)


if __name__ == "__main__":
    main()