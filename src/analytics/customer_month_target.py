import sqlite3
from contextlib import closing
from dataclasses import dataclass

from src.analytics.customer_month import (
    CustomerMonth,
    load_customer_months,
    validate_customer_months,
)
from src.engineering.config import KAGGLE_DATASETS


@dataclass(frozen=True)
class CustomerMonthTarget:
    customer_id: str
    reference_month: str
    target_active_next_month: int


def get_next_month(reference_month: str) -> str:
    year, month = map(
        int,
        reference_month.split("-"),
    )

    if month == 12:
        year += 1
        month = 1
    else:
        month += 1

    return f"{year:04d}-{month:02d}"


def build_customer_month_targets(
    customer_months: list[CustomerMonth],
) -> list[CustomerMonthTarget]:
    customer_month_index: dict[
        tuple[str, str],
        CustomerMonth,
    ] = {
        (
            customer_month.customer_id,
            customer_month.reference_month,
        ): customer_month
        for customer_month in customer_months
    }

    targets: list[CustomerMonthTarget] = []

    for customer_month in customer_months:
        next_month = get_next_month(
            customer_month.reference_month
        )

        next_month_key = (
            customer_month.customer_id,
            next_month,
        )

        next_customer_month = customer_month_index.get(
            next_month_key
        )

        if next_customer_month is None:
            continue

        target_active_next_month = int(
            next_customer_month.transaction_count > 0
        )

        targets.append(
            CustomerMonthTarget(
                customer_id=customer_month.customer_id,
                reference_month=customer_month.reference_month,
                target_active_next_month=(
                    target_active_next_month
                ),
            )
        )

    return targets


def validate_customer_month_targets(
    targets: list[CustomerMonthTarget],
) -> None:
    observed_keys: set[tuple[str, str]] = set()

    for target in targets:
        key = (
            target.customer_id,
            target.reference_month,
        )

        if key in observed_keys:
            raise ValueError(
                "Alvo duplicado para "
                f"{target.customer_id} em "
                f"{target.reference_month}."
            )

        observed_keys.add(key)

        if target.target_active_next_month not in (0, 1):
            raise ValueError(
                "Alvo inválido para "
                f"{target.customer_id} em "
                f"{target.reference_month}: "
                f"{target.target_active_next_month}."
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

    active_next_month = sum(
        target.target_active_next_month
        for target in targets
    )

    inactive_next_month = (
        len(targets)
        - active_next_month
    )

    active_rate = (
        active_next_month
        / len(targets)
    )

    print("Validação dos alvos: OK")
    print(f"Total de observações: {len(targets):,}")
    print(
        "Ativos no mês seguinte: "
        f"{active_next_month:,}"
    )
    print(
        "Inativos no mês seguinte: "
        f"{inactive_next_month:,}"
    )
    print(
        "Taxa de atividade no mês seguinte: "
        f"{active_rate:.2%}"
    )

    print("\nPrimeiros 10 alvos:")

    for target in targets[:10]:
        print(
            f"{target.customer_id} | "
            f"{target.reference_month} | "
            f"{target.target_active_next_month}"
        )


if __name__ == "__main__":
    main()