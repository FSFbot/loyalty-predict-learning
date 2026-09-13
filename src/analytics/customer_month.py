import sqlite3
from contextlib import closing
from dataclasses import dataclass

from src.engineering.config import KAGGLE_DATASETS


@dataclass(frozen=True)
class CustomerMonth:
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


def load_customer_months(
    connection: sqlite3.Connection,
) -> list[CustomerMonth]:
    connection.row_factory = sqlite3.Row

    query = """
        WITH RECURSIVE

        boundaries AS (
            SELECT
                date(
                    MAX(DtCriacao),
                    'start of month'
                ) AS last_month_start
            FROM transacoes
        ),

        first_transactions AS (
            SELECT
                IdCliente AS customer_id,
                MIN(DtCriacao) AS first_transaction_at
            FROM transacoes
            GROUP BY IdCliente
        ),

        customer_starts AS (
            SELECT
                customer.idCliente AS customer_id,
                CASE
                    WHEN first_transaction.first_transaction_at IS NULL
                    THEN customer.DtCriacao

                    WHEN datetime(
                        first_transaction.first_transaction_at
                    ) < datetime(customer.DtCriacao)
                    THEN first_transaction.first_transaction_at

                    ELSE customer.DtCriacao
                END AS effective_start_at
            FROM clientes AS customer
            LEFT JOIN first_transactions AS first_transaction
                ON first_transaction.customer_id = customer.idCliente
        ),

        eligible_customers AS (
            SELECT
                customer_id,
                date(
                    effective_start_at,
                    'start of month',
                    '+1 month'
                ) AS first_eligible_month
            FROM customer_starts
        ),

        calendar(reference_month) AS (
            SELECT
                MIN(first_eligible_month)
            FROM eligible_customers

            UNION ALL

            SELECT
                date(
                    calendar.reference_month,
                    '+1 month'
                )
            FROM calendar
            CROSS JOIN boundaries
            WHERE date(
                calendar.reference_month,
                '+1 month'
            ) < boundaries.last_month_start
        ),

        monthly_activity AS (
            SELECT
                transaction_data.IdCliente AS customer_id,
                date(
                    transaction_data.DtCriacao,
                    'start of month'
                ) AS reference_month,
                COUNT(*) AS transaction_count,
                SUM(
                    CASE
                        WHEN transaction_data.QtdePontos > 0
                        THEN 1
                        ELSE 0
                    END
                ) AS positive_transaction_count,
                SUM(
                    CASE
                        WHEN transaction_data.QtdePontos < 0
                        THEN 1
                        ELSE 0
                    END
                ) AS negative_transaction_count,
                SUM(
                    CASE
                        WHEN transaction_data.QtdePontos = 0
                        THEN 1
                        ELSE 0
                    END
                ) AS zero_point_transaction_count,
                SUM(
                    CASE
                        WHEN transaction_data.QtdePontos > 0
                        THEN transaction_data.QtdePontos
                        ELSE 0
                    END
                ) AS points_added,
                SUM(
                    CASE
                        WHEN transaction_data.QtdePontos < 0
                        THEN -transaction_data.QtdePontos
                        ELSE 0
                    END
                ) AS points_removed,
                SUM(
                    ABS(transaction_data.QtdePontos)
                ) AS points_moved,
                SUM(
                    transaction_data.QtdePontos
                ) AS net_points
            FROM transacoes AS transaction_data
            CROSS JOIN boundaries
            WHERE datetime(transaction_data.DtCriacao)
                  < boundaries.last_month_start
            GROUP BY
                transaction_data.IdCliente,
                reference_month
        )

        SELECT
            eligible_customer.customer_id,
            strftime(
                '%Y-%m',
                calendar.reference_month
            ) AS reference_month,
            COALESCE(
                monthly_activity.transaction_count,
                0
            ) AS transaction_count,
            COALESCE(
                monthly_activity.positive_transaction_count,
                0
            ) AS positive_transaction_count,
            COALESCE(
                monthly_activity.negative_transaction_count,
                0
            ) AS negative_transaction_count,
            COALESCE(
                monthly_activity.zero_point_transaction_count,
                0
            ) AS zero_point_transaction_count,
            COALESCE(
                monthly_activity.points_added,
                0
            ) AS points_added,
            COALESCE(
                monthly_activity.points_removed,
                0
            ) AS points_removed,
            COALESCE(
                monthly_activity.points_moved,
                0
            ) AS points_moved,
            COALESCE(
                monthly_activity.net_points,
                0
            ) AS net_points
        FROM eligible_customers AS eligible_customer
        INNER JOIN calendar
            ON calendar.reference_month
               >= eligible_customer.first_eligible_month
        LEFT JOIN monthly_activity
            ON monthly_activity.customer_id
               = eligible_customer.customer_id
           AND monthly_activity.reference_month
               = calendar.reference_month
        ORDER BY
            reference_month,
            eligible_customer.customer_id
    """

    cursor = connection.execute(query)
    rows = cursor.fetchall()

    return [
        CustomerMonth(
            customer_id=str(row["customer_id"]),
            reference_month=str(row["reference_month"]),
            transaction_count=int(row["transaction_count"]),
            positive_transaction_count=int(
                row["positive_transaction_count"]
            ),
            negative_transaction_count=int(
                row["negative_transaction_count"]
            ),
            zero_point_transaction_count=int(
                row["zero_point_transaction_count"]
            ),
            points_added=int(row["points_added"]),
            points_removed=int(row["points_removed"]),
            points_moved=int(row["points_moved"]),
            net_points=int(row["net_points"]),
        )
        for row in rows
    ]


def validate_customer_months(
    customer_months: list[CustomerMonth],
) -> None:
    for customer_month in customer_months:
        classified_transactions = (
            customer_month.positive_transaction_count
            + customer_month.negative_transaction_count
            + customer_month.zero_point_transaction_count
        )

        if (
            customer_month.transaction_count
            != classified_transactions
        ):
            raise ValueError(
                "Contagem de transações inconsistente para "
                f"{customer_month.customer_id} em "
                f"{customer_month.reference_month}."
            )

        expected_points_moved = (
            customer_month.points_added
            + customer_month.points_removed
        )

        if (
            customer_month.points_moved
            != expected_points_moved
        ):
            raise ValueError(
                "Volume de pontos inconsistente para "
                f"{customer_month.customer_id} em "
                f"{customer_month.reference_month}."
            )

        expected_net_points = (
            customer_month.points_added
            - customer_month.points_removed
        )

        if customer_month.net_points != expected_net_points:
            raise ValueError(
                "Saldo de pontos inconsistente para "
                f"{customer_month.customer_id} em "
                f"{customer_month.reference_month}."
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

    total_customer_months = len(customer_months)

    inactive_customer_months = sum(
        customer_month.transaction_count == 0
        for customer_month in customer_months
    )

    active_customer_months = (
        total_customer_months
        - inactive_customer_months
    )

    print("Validação das métricas: OK")
    print(
        "Total de combinações elegíveis: "
        f"{total_customer_months:,}"
    )
    print(
        "Meses com atividade: "
        f"{active_customer_months:,}"
    )
    print(
        "Meses sem atividade: "
        f"{inactive_customer_months:,}"
    )

    print("\nPrimeiros 5 objetos:")

    for customer_month in customer_months[:5]:
        print(customer_month)


if __name__ == "__main__":
    main()
