import sqlite3
from bisect import bisect_right
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
from src.analytics.historical_features import (
    HistoricalFeatureRow,
    build_historical_features,
    validate_historical_features,
)
from src.analytics.modeling_table import (
    build_modeling_table,
    validate_modeling_table,
)
from src.engineering.config import KAGGLE_DATASETS


@dataclass(frozen=True)
class RecencyFeatureRow:
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
    target_active_next_month: int


def load_active_months(
    connection: sqlite3.Connection,
) -> dict[str, list[str]]:
    connection.row_factory = sqlite3.Row

    query = """
        SELECT
            IdCliente AS customer_id,
            strftime(
                '%Y-%m',
                DtCriacao
            ) AS active_month
        FROM transacoes
        WHERE DtCriacao IS NOT NULL
          AND strftime(
                '%Y-%m',
                DtCriacao
              ) IS NOT NULL
        GROUP BY
            IdCliente,
            active_month
        ORDER BY
            customer_id,
            active_month
    """

    rows = connection.execute(query).fetchall()

    active_months_by_customer: defaultdict[
        str,
        list[str],
    ] = defaultdict(list)

    for row in rows:
        customer_id = str(row["customer_id"])
        active_month = str(row["active_month"])

        active_months_by_customer[customer_id].append(
            active_month
        )

    return dict(active_months_by_customer)


def calculate_month_distance(
    earlier_month: str,
    later_month: str,
) -> int:
    earlier_year, earlier_month_number = map(
        int,
        earlier_month.split("-"),
    )
    later_year, later_month_number = map(
        int,
        later_month.split("-"),
    )

    return (
        (later_year - earlier_year) * 12
        + later_month_number
        - earlier_month_number
    )


def build_recency_features(
    historical_rows: list[HistoricalFeatureRow],
    active_months_by_customer: dict[str, list[str]],
) -> list[RecencyFeatureRow]:
    recency_rows: list[RecencyFeatureRow] = []

    for row in historical_rows:
        customer_active_months = (
            active_months_by_customer.get(
                row.customer_id,
                [],
            )
        )

        insertion_position = bisect_right(
            customer_active_months,
            row.reference_month,
        )

        if insertion_position == 0:
            months_since_last_activity = None
        else:
            last_active_month = customer_active_months[
                insertion_position - 1
            ]

            months_since_last_activity = (
                calculate_month_distance(
                    last_active_month,
                    row.reference_month,
                )
            )

        recency_rows.append(
            RecencyFeatureRow(
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
                    months_since_last_activity
                ),
                target_active_next_month=(
                    row.target_active_next_month
                ),
            )
        )

    return recency_rows


def validate_recency_features(
    recency_rows: list[RecencyFeatureRow],
    historical_rows: list[HistoricalFeatureRow],
) -> None:
    if not recency_rows:
        raise ValueError(
            "A tabela de referencia esta¡ vazia."
        )

    if len(recency_rows) != len(historical_rows):
        raise ValueError(
            "A quantidade de linhas mudou durante o calculo "
            "da referencia."
        )

    expected_keys = {
        (
            row.customer_id,
            row.reference_month,
        )
        for row in historical_rows
    }

    observed_keys: set[tuple[str, str]] = set()

    for row in recency_rows:
        key = (
            row.customer_id,
            row.reference_month,
        )

        if key in observed_keys:
            raise ValueError(
                "Referencia duplicada para "
                f"{row.customer_id} em "
                f"{row.reference_month}."
            )

        observed_keys.add(key)

        if (
            row.months_since_last_activity is not None
            and row.months_since_last_activity < 0
        ):
            raise ValueError(
                "Referencia negativa para "
                f"{row.customer_id} em "
                f"{row.reference_month}."
            )

        if (
            row.transaction_count > 0
            and row.months_since_last_activity != 0
        ):
            raise ValueError(
                "Cliente ativo no mais atual com referencia "
                "diferente de zero: "
                f"{row.customer_id} em "
                f"{row.reference_month}."
            )

        if (
            row.transaction_count == 0
            and row.months_since_last_activity == 0
        ):
            raise ValueError(
                "Cliente inativo no mais atual com recÃªncia zero: "
                f"{row.customer_id} em "
                f"{row.reference_month}."
            )

    if observed_keys != expected_keys:
        raise ValueError(
            "As chaves foram alteradas durante o calculo "
            "da recÃªncia."
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

    no_known_activity = sum(
        row.months_since_last_activity is None
        for row in recency_rows
    )

    active_current_month = sum(
        row.months_since_last_activity == 0
        for row in recency_rows
    )

    inactive_with_history = (
        len(recency_rows)
        - no_known_activity
        - active_current_month
    )

    known_recencies = [
        row.months_since_last_activity
        for row in recency_rows
        if row.months_since_last_activity is not None
    ]

    maximum_recency = max(
        known_recencies,
        default=0,
    )

    print("Validação das caracterÃ­sticas de recÃªncia: OK")
    print(f"Total de linhas: {len(recency_rows):,}")
    print(
        "Sem atividade conhecida: "
        f"{no_known_activity:,}"
    )
    print(
        "Ativos no mais atual: "
        f"{active_current_month:,}"
    )
    print(
        "Inativos com atividade anterior: "
        f"{inactive_with_history:,}"
    )
    print(
        "Maior referencia observada: "
        f"{maximum_recency} meses"
    )

    print("\nPrimeiros 5 clientes inativos com histÃ³rico:")

    inactive_examples = [
        row
        for row in recency_rows
        if row.transaction_count == 0
        and row.months_since_last_activity is not None
    ]

    for row in inactive_examples[:5]:
        print(row)


if __name__ == "__main__":
    main()
