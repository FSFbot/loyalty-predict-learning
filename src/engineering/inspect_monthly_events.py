import sqlite3

from .config import KAGGLE_DATASETS
from .inspect_time_ranges import (
    CHECKS,
    TemporalCheck,
    quote_identifier,
)


def count_events_by_month(
    connection: sqlite3.Connection,
    check: TemporalCheck,
) -> list[tuple[str, int]]:
    table = quote_identifier(check.table)
    date_column = quote_identifier(check.date_column)

    query = f"""
        SELECT
            strftime('%Y-%m', {date_column}) AS event_month,
            COUNT(*) AS total_events
        FROM {table}
        WHERE {date_column} IS NOT NULL
          AND datetime({date_column}) IS NOT NULL
        GROUP BY event_month
        ORDER BY event_month
    """

    rows = connection.execute(query).fetchall()

    return [
        (row[0], int(row[1]))
        for row in rows
    ]


def main() -> None:
    databases = {
        dataset.destination.name: dataset.destination / "database.db"
        for dataset in KAGGLE_DATASETS
    }

    for source_name, checks in CHECKS.items():
        print(f"\nFonte: {source_name}")

        with sqlite3.connect(databases[source_name]) as connection:
            for check in checks:
                monthly_counts = count_events_by_month(
                    connection,
                    check,
                )

                print(f"\nTabela: {check.table}")

                for event_month, total_events in monthly_counts:
                    print(f"  {event_month}: {total_events:,}")


if __name__ == "__main__":
    main()