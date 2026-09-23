import sqlite3
from collections import defaultdict
from contextlib import closing
from dataclasses import dataclass

from src.analytics.customer_month import (
    load_customer_months,
    validate_customer_months,
)
from src.analytics.customer_month_target import (
    build_customer_month_targets,
    validate_customer_month_targets,
)
from src.analytics.modeling_table import (
    ModelingRow,
    build_modeling_table,
    validate_modeling_table,
)
from src.engineering.config import KAGGLE_DATASETS


@dataclass(frozen=True)
class HistoricalFeatureRow:
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
    target_active_next_month: int


def build_historical_features(
    modeling_rows: list[ModelingRow],
) -> list[HistoricalFeatureRow]:
    rows_by_customer: defaultdict[
        str,
        list[ModelingRow],
    ] = defaultdict(list)

    for row in modeling_rows:
        rows_by_customer[row.customer_id].append(row)

    historical_rows: list[HistoricalFeatureRow] = []

    for customer_rows in rows_by_customer.values():
        customer_rows.sort(
            key=lambda row: row.reference_month
        )

        for row_index, current_row in enumerate(
            customer_rows
        ):
            window_start = max(
                0,
                row_index - 2,
            )

            window_rows = customer_rows[
                window_start : row_index + 1
            ]

            observed_months_last_3 = len(window_rows)

            active_months_last_3 = sum(
                window_row.transaction_count > 0
                for window_row in window_rows
            )

            transaction_count_last_3_months = sum(
                window_row.transaction_count
                for window_row in window_rows
            )

            net_points_last_3_months = sum(
                window_row.net_points
                for window_row in window_rows
            )

            historical_rows.append(
                HistoricalFeatureRow(
                    customer_id=current_row.customer_id,
                    reference_month=(
                        current_row.reference_month
                    ),
                    transaction_count=(
                        current_row.transaction_count
                    ),
                    positive_transaction_count=(
                        current_row.positive_transaction_count
                    ),
                    negative_transaction_count=(
                        current_row.negative_transaction_count
                    ),
                    zero_point_transaction_count=(
                        current_row.zero_point_transaction_count
                    ),
                    points_added=current_row.points_added,
                    points_removed=current_row.points_removed,
                    points_moved=current_row.points_moved,
                    net_points=current_row.net_points,
                    observed_months_last_3=(
                        observed_months_last_3
                    ),
                    active_months_last_3=(
                        active_months_last_3
                    ),
                    transaction_count_last_3_months=(
                        transaction_count_last_3_months
                    ),
                    net_points_last_3_months=(
                        net_points_last_3_months
                    ),
                    target_active_next_month=(
                        current_row.target_active_next_month
                    ),
                )
            )

    historical_rows.sort(
        key=lambda row: (
            row.reference_month,
            row.customer_id,
        )
    )

    return historical_rows


def validate_historical_features(
    historical_rows: list[HistoricalFeatureRow],
    modeling_rows: list[ModelingRow],
) -> None:
    if not historical_rows:
        raise ValueError(
            "A tabela de características históricas está vazia."
        )

    if len(historical_rows) != len(modeling_rows):
        raise ValueError(
            "A quantidade de linhas mudou durante a construção "
            "das características históricas."
        )

    expected_keys = {
        (
            row.customer_id,
            row.reference_month,
        )
        for row in modeling_rows
    }

    observed_keys: set[tuple[str, str]] = set()

    for row in historical_rows:
        key = (
            row.customer_id,
            row.reference_month,
        )

        if key in observed_keys:
            raise ValueError(
                "Característica histórica duplicada para "
                f"{row.customer_id} em "
                f"{row.reference_month}."
            )

        observed_keys.add(key)

        if not 1 <= row.observed_months_last_3 <= 3:
            raise ValueError(
                "Quantidade inválida de meses observados para "
                f"{row.customer_id} em "
                f"{row.reference_month}."
            )

        if not (
            0
            <= row.active_months_last_3
            <= row.observed_months_last_3
        ):
            raise ValueError(
                "Quantidade inválida de meses ativos para "
                f"{row.customer_id} em "
                f"{row.reference_month}."
            )

        if (
            row.transaction_count_last_3_months
            < row.transaction_count
        ):
            raise ValueError(
                "A soma histórica de transações é menor que "
                "a contagem do mês atual para "
                f"{row.customer_id} em "
                f"{row.reference_month}."
            )

        if row.target_active_next_month not in (0, 1):
            raise ValueError(
                "Alvo inválido para "
                f"{row.customer_id} em "
                f"{row.reference_month}."
            )

    if observed_keys != expected_keys:
        raise ValueError(
            "As chaves foram alteradas durante a construção "
            "das características históricas."
        )


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

    full_window_rows = sum(
        row.observed_months_last_3 == 3
        for row in historical_rows
    )

    partial_window_rows = (
        len(historical_rows)
        - full_window_rows
    )

    print("Validação das características históricas: OK")
    print(f"Total de linhas: {len(historical_rows):,}")
    print(
        "Linhas com janela completa: "
        f"{full_window_rows:,}"
    )
    print(
        "Linhas com janela parcial: "
        f"{partial_window_rows:,}"
    )

    print("\nPrimeiras 5 janelas completas:")

    complete_examples = [
        row
        for row in historical_rows
        if row.observed_months_last_3 == 3
    ]

    for row in complete_examples[:5]:
        print(row)


if __name__ == "__main__":
    main()