import sqlite3
from contextlib import closing
from dataclasses import dataclass

from src.analytics.customer_month import (
    CustomerMonth,
    load_customer_months,
    validate_customer_months,
)
from src.analytics.customer_month_target import (
    CustomerMonthTarget,
    build_customer_month_targets,
    validate_customer_month_targets,
)
from src.engineering.config import KAGGLE_DATASETS


@dataclass(frozen=True)
class ModelingRow:
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
    target_active_next_month: int


def build_modeling_table(
    customer_months: list[CustomerMonth],
    targets: list[CustomerMonthTarget],
) -> list[ModelingRow]:
    target_index: dict[tuple[str, str], int] = {
        (
            target.customer_id,
            target.reference_month,
        ): target.target_active_next_month
        for target in targets
    }

    modeling_rows: list[ModelingRow] = []

    for customer_month in customer_months:
        key = (
            customer_month.customer_id,
            customer_month.reference_month,
        )

        target_value = target_index.get(key)

        if target_value is None:
            continue

        modeling_rows.append(
            ModelingRow(
                customer_id=customer_month.customer_id,
                reference_month=(
                    customer_month.reference_month
                ),
                transaction_count=(
                    customer_month.transaction_count
                ),
                positive_transaction_count=(
                    customer_month.positive_transaction_count
                ),
                negative_transaction_count=(
                    customer_month.negative_transaction_count
                ),
                zero_point_transaction_count=(
                    customer_month.zero_point_transaction_count
                ),
                points_added=customer_month.points_added,
                points_removed=customer_month.points_removed,
                points_moved=customer_month.points_moved,
                net_points=customer_month.net_points,
                target_active_next_month=target_value,
            )
        )

    return modeling_rows


def validate_modeling_table(
    modeling_rows: list[ModelingRow],
    targets: list[CustomerMonthTarget],
) -> None:
    if not modeling_rows:
        raise ValueError(
            "A tabela de modelagem está vazia."
        )

    expected_keys = {
        (
            target.customer_id,
            target.reference_month,
        )
        for target in targets
    }

    observed_keys: set[tuple[str, str]] = set()

    for row in modeling_rows:
        key = (
            row.customer_id,
            row.reference_month,
        )

        if key in observed_keys:
            raise ValueError(
                "Linha de modelagem duplicada para "
                f"{row.customer_id} em "
                f"{row.reference_month}."
            )

        observed_keys.add(key)

        if row.target_active_next_month not in (0, 1):
            raise ValueError(
                "Alvo inválido na tabela de modelagem para "
                f"{row.customer_id} em "
                f"{row.reference_month}: "
                f"{row.target_active_next_month}."
            )

    if observed_keys != expected_keys:
        missing_keys = expected_keys - observed_keys
        unexpected_keys = observed_keys - expected_keys

        raise ValueError(
            "As chaves da tabela de modelagem não correspondem "
            "às chaves dos alvos. "
            f"Ausentes: {len(missing_keys):,}. "
            f"Inesperadas: {len(unexpected_keys):,}."
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

    active_next_month = sum(
        row.target_active_next_month
        for row in modeling_rows
    )

    inactive_next_month = (
        len(modeling_rows)
        - active_next_month
    )

    print("Validação da tabela de modelagem: OK")
    print(f"Total de linhas: {len(modeling_rows):,}")
    print(
        "Ativos no mês seguinte: "
        f"{active_next_month:,}"
    )
    print(
        "Inativos no mês seguinte: "
        f"{inactive_next_month:,}"
    )
    print(
        "Período de referência: "
        f"{min(row.reference_month for row in modeling_rows)} "
        "até "
        f"{max(row.reference_month for row in modeling_rows)}"
    )

    print("\nPrimeiras 5 linhas:")

    for row in modeling_rows[:5]:
        print(row)


if __name__ == "__main__":
    main()