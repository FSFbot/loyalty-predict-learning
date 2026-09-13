import sqlite3
from contextlib import closing

from src.engineering.config import KAGGLE_DATASETS


def load_customer_months(
    connection: sqlite3.Connection,
) -> list[tuple[str, str, int, int]]:
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
                COALESCE(
                    SUM(transaction_data.QtdePontos),
                    0
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
        (
            str(row[0]),
            str(row[1]),
            int(row[2]),
            int(row[3]),
        )
        for row in rows
    ]


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

    total_customer_months = len(customer_months)

    inactive_customer_months = sum(
        transaction_count == 0
        for (
            _,
            _,
            transaction_count,
            _,
        ) in customer_months
    )

    active_customer_months = (
        total_customer_months
        - inactive_customer_months
    )

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

    print("\nPrimeiras 10 linhas:")

    for (
        customer_id,
        reference_month,
        transaction_count,
        net_points,
    ) in customer_months[:10]:
        print(
            customer_id,
            reference_month,
            transaction_count,
            net_points,
            sep=" | ",
        )


if __name__ == "__main__":
    main()