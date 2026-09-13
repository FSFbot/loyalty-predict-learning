import sqlite3
from contextlib import closing

from src.engineering.config import KAGGLE_DATASETS


def load_active_customer_months(
    connection: sqlite3.Connection,
) -> list[tuple[str, str, int, int]]:
    query = """
        WITH boundaries AS (
            SELECT
                date(
                    MIN(DtCriacao),
                    'start of month',
                    '+1 month'
                ) AS first_full_month,
                date(
                    MAX(DtCriacao),
                    'start of month'
                ) AS last_month_start
            FROM transacoes
        )
        SELECT
            transaction_data.IdCliente AS customer_id,
            strftime(
                '%Y-%m',
                transaction_data.DtCriacao
            ) AS reference_month,
            COUNT(*) AS transaction_count,
            COALESCE(
                SUM(transaction_data.QtdePontos),
                0
            ) AS net_points
        FROM transacoes AS transaction_data

        INNER JOIN clientes AS customer
            ON customer.idCliente = transaction_data.IdCliente

        CROSS JOIN boundaries

        WHERE datetime(transaction_data.DtCriacao)
              >= boundaries.first_full_month
          AND datetime(transaction_data.DtCriacao)
              < boundaries.last_month_start

        GROUP BY
            transaction_data.IdCliente,
            reference_month

        ORDER BY
            reference_month,
            transaction_data.IdCliente
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
        customer_months = load_active_customer_months(
            connection
        )

    print(
        "Combinações ativas de cliente e mês: "
        f"{len(customer_months):,}"
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